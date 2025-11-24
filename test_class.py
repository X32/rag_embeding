class Car:
    """
    汽车类：封装汽车的核心属性和行为
    支持：启动、加速、刹车、熄火、显示状态等操作
    """

    def __init__(self, brand: str, model: str, year: int, color: str, max_speed: int = 220):
        """
        构造方法：初始化汽车的核心属性
        :param brand: 汽车品牌（如 "Toyota"、"BMW"）
        :param model: 汽车型号（如 "Camry"、"3 Series"）
        :param year: 生产年份（如 2023）
        :param color: 车身颜色（如 "white"、"black"）
        :param max_speed: 最高时速（默认 220 km/h，可选参数）
        """
        # 公共属性（可直接访问）
        self.brand = brand  # 品牌
        self.model = model  # 型号
        self.year = year  # 生产年份
        self.color = color  # 颜色

        # 私有属性（通过方法访问，避免直接修改）
        self._max_speed = max_speed  # 最高时速
        self._current_speed = 0  # 当前速度（初始为 0）
        self._is_running = False  # 运行状态（初始为熄火）

    # ---------------------- 公共方法：汽车操作 ----------------------
    def start(self) -> None:
        """启动汽车"""
        if not self._is_running:
            self._is_running = True
            print(f"✅ {self._get_full_name()} 启动成功！当前速度：{self._current_speed} km/h")
        else:
            print(f"⚠️  提示：{self._get_full_name()} 已处于启动状态，无需重复启动")

    def accelerate(self, speed_increase: int) -> None:
        """
        加速操作
        :param speed_increase: 增加的速度（正数）
        """
        if not self._is_running:
            print(f"❌ 错误：{self._get_full_name()} 未启动，请先启动汽车")
            return

        if speed_increase <= 0:
            print(f"❌ 错误：加速速度必须为正数（当前输入：{speed_increase}）")
            return

        # 加速后速度不能超过最高时速
        new_speed = self._current_speed + speed_increase
        self._current_speed = min(new_speed, self._max_speed)
        print(f"⚡ {self._get_full_name()} 加速 {speed_increase} km/h，当前速度：{self._current_speed} km/h")

    def brake(self, speed_decrease: int) -> None:
        """
        刹车操作（减速）
        :param speed_decrease: 减少的速度（正数）
        """
        if not self._is_running:
            print(f"❌ 错误：{self._get_full_name()} 未启动，无需刹车")
            return

        if speed_decrease <= 0:
            print(f"❌ 错误：刹车速度必须为正数（当前输入：{speed_decrease}）")
            return

        # 刹车后速度不能低于 0
        new_speed = self._current_speed - speed_decrease
        self._current_speed = max(new_speed, 0)
        print(f"🛑 {self._get_full_name()} 刹车减速 {speed_decrease} km/h，当前速度：{self._current_speed} km/h")

    def stop(self) -> None:
        """熄火汽车（会先将速度归零）"""
        if self._is_running:
            self._current_speed = 0  # 熄火前强制停车
            self._is_running = False
            print(f"✅ {self._get_full_name()} 已熄火！当前速度：{self._current_speed} km/h")
        else:
            print(f"⚠️  提示：{self._get_full_name()} 已处于熄火状态，无需重复熄火")

    def get_status(self) -> str:
        """获取汽车当前状态（字符串描述）"""
        status = "运行中" if self._is_running else "已熄火"
        return (
            f"🚗 汽车状态：\n"
            f"  品牌型号：{self.brand} {self.model}\n"
            f"  生产年份：{self.year}年\n"
            f"  车身颜色：{self.color}\n"
            f"  最高时速：{self._max_speed} km/h\n"
            f"  当前状态：{status}\n"
            f"  当前速度：{self._current_speed} km/h"
        )

    # ---------------------- 私有方法：内部辅助 ----------------------
    def _get_full_name(self) -> str:
        """私有方法：获取汽车完整名称（品牌+型号+年份）"""
        return f"{self.year}款 {self.brand} {self.model}"

    # ---------------------- 特殊方法：重载内置行为 ----------------------
    def __str__(self) -> str:
        """重载 print() 输出：返回汽车简洁描述"""
        return f"{self._get_full_name()}（{self.color}，最高时速{self._max_speed}km/h）"

    def __repr__(self) -> str:
        """重载控制台直接输出：返回可复用的实例构造字符串"""
        return f"Car(brand='{self.brand}', model='{self.model}', year={self.year}, color='{self.color}', max_speed={self._max_speed})"


# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":
    # 1. 创建汽车实例（丰田凯美瑞）
    camry = Car(brand="Toyota", model="Camry", year=2024, color="珍珠白", max_speed=210)
    print("=== 初始化汽车 ===")
    print(camry)  # 调用 __str__ 方法
    print(repr(camry))  # 调用 __repr__ 方法

    # 2. 测试汽车操作
    print("\n=== 测试汽车功能 ===")
    camry.accelerate(50)  # 未启动时加速（错误）
    camry.start()  # 启动汽车
    camry.accelerate(50)  # 加速 50 km/h
    camry.accelerate(80)  # 再加速 80 km/h（当前速度 130）
    camry.accelerate(100)  # 尝试加速 100 km/h（超过最高速 210，会被限制）
    camry.brake(30)  # 刹车减速 30 km/h
    camry.brake(120)  # 再刹车减速 120 km/h（速度归零）

    # 3. 查看汽车状态
    print("\n=== 查看汽车状态 ===")
    print(camry.get_status())

    # 4. 熄火汽车
    print("\n=== 熄火汽车 ===")
    camry.stop()
    camry.stop()  # 重复熄火（提示）

    # 5. 创建第二个汽车实例（宝马3系）
    print("\n=== 创建第二个汽车实例 ===")
    bmw = Car(brand="BMW", model="3 Series", year=2025, color="曜夜黑", max_speed=250)
    bmw.start()
    bmw.accelerate(120)
    print(bmw.get_status())