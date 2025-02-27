from .abstract_driver_infinity import AbstractDriverInfinity
from netket.driver.vmc import VMC

class VMCInfinity(AbstractDriverInfinity, VMC):
    # 如果VMC有__init__方法需要特殊处理，可能需要重写
    def __init__(self, *args, **kwargs):
        VMC.__init__(self, *args, **kwargs)  # 直接调用VMC的初始化
        AbstractDriverInfinity.__init__(self, *args, **kwargs)