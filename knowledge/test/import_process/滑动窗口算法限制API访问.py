import time
from collections import deque


class RateLimiter:
    def __init__(self, requests_per_minute: int, window_seconds: int = 60):
        self.max_req = requests_per_minute  # 每分钟最大请求数
        self.window = window_seconds  # 窗口60秒
        self.timestamps: deque[float] = deque()

    def wait(self):
        """执行前调用，自动限流等待"""
        now = time.time()
        # 步骤1：清理窗口外过期的时间戳
        while self.timestamps and now - self.timestamps[0] >= self.window:
            self.timestamps.popleft()

        # 步骤2：达到上限，需要等待
        if len(self.timestamps) >= self.max_req:
            # 算出还要等多久
            wait_sec = self.window - (now - self.timestamps[0])
            if wait_sec > 0:
                print(f"触发限流，等待 {wait_sec:.2f} 秒")
                time.sleep(wait_sec)
            # 等待完再清一次过期数据
            now = time.time()
            while self.timestamps and now - self.timestamps[0] >= self.window:
                self.timestamps.popleft()

        # 步骤3：记录本次请求时间
        self.timestamps.append(time.time())


# 测试示例：1分钟最多3次请求
if __name__ == "__main__":
    limiter = RateLimiter(requests_per_minute=3)
    for i in range(11):
        print(f"\n准备第{i + 1}次请求")
        limiter.wait()
        print(f"执行第{i + 1}次请求")