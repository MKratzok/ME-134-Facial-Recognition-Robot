from adafruit_servokit import ServoKit
from time import sleep

''' PIN MAP:
    [0] is tail
    [1] is mid-tail
    [2] is hip
    [3] is mid-head
    [4] is head
    [5] is suction cup
'''

EXTEND = [65, 100, 90, 110, 65, -1]
CURL = [170, 0, -1, 0, 159, -1]

class Robot:
    def __init__(self):
        self.cam = camera.Camera()
        self.kit = ServoKit(channels=16)
        self._unstick()
        self._set_all(EXTEND)

        self.stuck = False

    def _set_all(self, arr):
        """
        _set_all(arr) - sets all servos to the values in the array
        :param arr: the array of ints for the servos angles. Use -1 to skip value
        :return: none
        """
        for i in range(0,len(arr)):
            if arr[i] == -1:
                continue
            else:
                self.kit.servo[i].angle = arr[i]

    def _stick(self):
        """
        _stick() - sticks the suction cup
        :return: none
        """
        self.kit.servo[5].angle = 60
        self.stuck = True

    def _unstick(self):
        """
        _unstick() - unsticks the suction cup
        :return: none
        """
        self.kit.servo[5].angle = 130
        self.stuck = False

    def _curl_up(self):
        """
        _curl_up() - arches the inchworm robot's back
        :return: none
        """
        self._set_all(CURL)

    def _extend(self):
        """
        _extend - straightens the inchworm robot's back
        :return: none
        """
        self.kit.servo[0].angle = 120
        self.kit.servo[1].angle = 60
        sleep(0.15)
        self.kit.servo[3].angle = 20
        self.kit.servo[4].angle = 130
        sleep(0.15)
        self.kit.servo[3].angle = 60
        self.kit.servo[4].angle = 100
        sleep(0.15)
        self._set_all(EXTEND)

    def crawl(self):
        """
        crawl() - executes the crawl sequence for the robot
        :return: none
        """
        self._stick()
        sleep(0.5)
        self._curl_up()
        sleep(0.5)
        self._unstick()
        sleep(0.5)
        self._extend()
        sleep(0.5)

    def dance(self, cycles):
        """
        dance() - does a little dance
        :param cycles: the number of dance wiggles to do
        :return: none
        """
        for i in range(0,cycles):
            self.kit.servo[2].angle = 135
            sleep(0.5)
            self.kit.servo[2].angle = 90
            sleep(0.1)
            self.kit.servo[2].angle = 45
            sleep(0.5)
            self.kit.servo[2].angle = 90
            sleep(0.1)

    def turn(self, degrees):
        """
        turn(degrees) - turns the robot in maximum steps of 45 degrees
        :param degrees: the total number of degrees to turn
        :return: none
        """
        while degrees > 90 or degrees < -90:
            if degrees > 90:
                self.turn(90)
                degrees -= 90
            elif degrees < -90:
                self.turn(-90)
                degrees += 90

        self.kit.servo[2].angle = 90 + degrees
        sleep(0.5)
        self._stick()
        sleep(0.5)
        self.kit.servo[2].angle = 90
        sleep(0.5)
        self._unstick()
        sleep(0.5)

    def suction(self):
        if self.stuck:
            self._unstick()
        else:
            self._stick()

    def sees_blue(self):
        return self.cam.sees_blue()