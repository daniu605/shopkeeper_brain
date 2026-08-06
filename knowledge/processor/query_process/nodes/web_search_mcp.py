"""MCP 网络搜索节点

通过 MCP 协议调用网络搜索服务获取外部信息。
"""

import asyncio
import json
import logging
import asyncio
import os


from agents.mcp import MCPServerStreamableHttp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    """MCP 网络搜索节点。

    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
    MCP 形式调用第三方的各种通用的搜索工具。
    """

    name = "mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行网络搜索。

        Args:
            state: 查询图状态，需包含 rewritten_query 和 item_names。

        Returns:
            更新后的状态，包含 web_search_docs 搜索结果。
        """
        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具 bailian_web_search
        self.log_step("步骤2", f"MCP 网络搜索: {validated_rewritten_query[:50]}...")
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            self.logger.warning("MCP 网络搜索未返回结果")
            return state

        # 3. 只更新 state 的 web_search_docs
        self.logger.info(f"MCP 网络搜索完成，获取 {len(mcp_result)} 条结果")
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数。

        Args:
            state: 查询图状态。

        Returns:
            校验后的 rewritten_query 和 item_names。

        Raises:
            StateFieldError: 参数校验失败时抛出。
        """
        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        self.log_step("步骤1", "参数校验通过")
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """异步执行 MCP 网络搜索。

        Args:
            validated_rewritten_query: 校验后的查询文本。

        Returns:
            搜索结果列表，每项包含 title、url、snippet。
        """
        # 1. 创建 MCP 客户端

        self.logger.info("创建 MCP 客户端")
        async with MCPServerStreamableHttp(
                name="通用搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                    "timeout": 20,
                },
                cache_tools_list=True,
                max_retry_attempts=2
            )as client:
            # 2. 调用工具
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_rewritten_query, "count": self.config.mcp_search_limit}
            )

            # 4. 解析工具执行完的结果
            if not execute_tool_result:
                self.logger.warning("MCP 工具执行结果为空")
                return []
            if not execute_tool_result.content[0]:
                self.logger.warning("MCP 工具结果 content 为空")
                return []

            # 4.1 获取 TextContent 对象的 text
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                self.logger.warning("MCP 工具结果 text 为空")
                return []

            # 4.2 反序列化
            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                # a) 获取 pages
                pages = text_content_text.get('pages', "")
                if not pages:
                    self.logger.warning("MCP 结果中无 pages 字段")
                    return []

                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})

                # c) 最终返回
                self.logger.info(f"MCP 搜索结果解析完成，共 {len(search_result)} 条")
                return search_result

            except Exception as e:
                self.logger.error(f"反序列化 MCP 结果失败: {e}")
                return []




if __name__ == '__main__':
    state = {
        "rewritten_query": "介绍东微智能科技有限公司",
        "item_names": ["RS-12 数字万用表"]
    }

    mcp_search = McpSearchNode()
    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))







"""
{
  "snippet": "深圳市东微智能科技股份有限公司 100-499人 计算机软件 广东省 公司简介 东微是深圳市东微智能科技股份有限公司旗下品牌。 深圳市东微智能科技股份有限公司(TENDZONE)成立于2010年,是一家集研发、生产、销售及技术服务于一体的国家级高新技术企业,国家级专精特新重点“小巨人”,深圳市知名品牌。公司专注于视频、音频、控制、数据等流媒体的数字化、网络化、智能化技术的研究,用通信理念研发创新性的产品,产品覆盖数字音频、分布式网络音频、AI智能语音、智能显示与呈现、全媒体可视化集控、融合通讯、物联通信、物联数据等领域。东微智能具备多媒体通讯、视频、音频、控制等领域的完整底层核心技术与知识产权,目前已拥有200多项专利及著作权。核心团队来自于通信多媒体领域国际一流科技公司,相关人员均具备15年以上该领域行业经验及专业知识。 2010年TENDZONE数字音频处理器在国内成功发布后,彻底打破了海外厂家的垄断地位;2013年在行业内率先提出了视、音频处理分布式的概念,并推出了基于分布式架构的多媒体交互系统(MIDIS),全球首家实现从底层打通多媒体应用业务、监控系统、视频会议系统,有着丰富对接应用案例的全媒体融合管控平台。目前MIDIS系统已经发展到CloudMIDIS云媒体时代,东微不断锐意进取,持续处于行业领先地位。 公司稳健发展,业务遍布覆盖多个行业和领域,在公检法司、军民融合、应急、教育、气象、人防、公共安全、智能建筑、智慧城市、文旅、大数据等领域具有广泛成功案例。多年以来,TENDZONE系统多次入选国内外重大项目,在上合组织首脑峰会、武汉军运会、亚投行、国家财政部、中国天眼、安徽省司法厅、交通银行、新浪大厦、北京大学、俄罗斯联邦海关学院、29届世界大运会等项目中得到广泛成功应用。 工商信息 深圳市东微智能科技股份有限公司 法定代表人: 季海交  注册资本: 5326.163万人民币  成立日期: 2010-12-21  经营状态: 存续(在营、开业、在册)  注册地址: 深圳市南山区西丽街道西丽社区兴科一街万科云城一期七栋A座1601研发用房(国际创新谷1栋A座16层)  统一社会信用代码: 91440300567051204L  经营范围: 系统集成工程;经济信息咨询,企业管理咨询,电子技术、通讯技术、计算机技术的开发、技术转让、技术咨询;计算机网络工程、通讯器材、电子产品、音响器材、办公设备的销售;展览展示策划;国内贸易;经营进出口业务。(以上均不含法律、行政法规、国务院决定规定需前置审批和禁止的项目);广播电视设备制造(不含广播电视传输设",
  "title": "深圳市东微智能科技股份有限公司",
  "url": "https://www.zhipin.com/companys/1b76db03938f8a431nZ_0ti6Eg~~.html"
}
{
  "snippet": "深圳市东微智能科技股份有限公司 深圳市东微智能科技股份有限公司 公司概况about us 深圳市东微智能科技股份有限公司(东微TENDZONE)是专注于视音频流媒体的数字化、网络化、智能化研究集研发、生产、销售及技术服务的国家高新企业。东微产品覆盖了数字音频、分布式网络音频、AI智能语音、智能显示与呈现、全媒体可视化集控、融合通讯、物联通信、物联数据等领域,涉及多媒体信息分布式交、数字音频处理器、拼接融合处理器、录播机、信号矩阵、专业音箱、扬声器、功放、麦克风、数字调音台、数字会议等。 联系我们contact us 深圳市南山区打石1路深圳国际创新谷1栋A座16层、17层 (联系就说在DAV音视工程网上看到的) sales@tendzone.com;service@tendzone.com 0755-33991098 东微Tendzone产品",
  "title": "数字音视·企业厂商展示平台深圳市东微智能科技股份有限公司",
  "url": "https://tendzone.corp.dav01.com/"
}
{
  "snippet": "深圳市东微智能科技股份有限公司 已认领 开业 a级纳税人(2025) 司法案件 涉诉关系 高新技术企业(2025) 小微企业 在售商品 法定代表人: 季海交 注册资本: 5,326.163万(元) 统一社会信用代码: 91440300567051204l 电话: 0755-3399**** 邮箱: h****@tendzone.com 官网: - 地址: 深圳市南山区西丽街道西丽社区兴科一街万科云城一期七栋a座1601研发用房(国际创新谷1栋a座16层) 简介: 1,基本情况 深圳市东微智能科技股份有限公司是一家专精特新中小企业,高新技术企业(2024),小微企业,该公司成立于2010年12月21日,位于深圳市南山区西丽街道西丽社区兴科一街万科云城一期七栋a座1601研发用房(国际创新谷1栋a座16层),目前处于开业状态,经营范围包括系统集成工程;经济信息咨询,企业管理咨询,电子技术,通讯技术,计算机技术的开发,技术转让,技术咨询;计算机网络工程,通讯器材,电子产品,音响器材,办公设备的销售;展览展示策划;国内贸易;经营进出口业务等,企业品牌项目包括东微智能. 2,人员情况 深圳市东微智能科技股份有限公司法定代表人为季海交,参保人数为164人,其中季海交担任董事长,总经理,曾维雄担任董事,何茂盛担任董事,孟宪富担任董事,郝明军担任董事,罗斌担任监事,黄维担任监事,阮龙年担任监事. 3,知识产权 深圳市东微智能科技股份有限公司创新能力卓越,拥有雄厚的知识产权实力,其中包括1个企业品牌项目,23个注册商标,216个专利信息,142个软件著作权. 4,疑似负面信息方面 该公司曾涉及16件立案信息,5个法院公告,19个开庭公告,12起涉诉关系. 上述文章是根据发布当时全网最新公开信息生成的,仅供您参考,如需了解该企业最新及更全面数据可前往 爱企查 查看.",
  "title": "深圳市东微智能科技股份有限公司",
  "url": "https://aiqicha.baidu.com/company_detail_11839113935937"
}
"""