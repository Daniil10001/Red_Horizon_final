import numpy as np
import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import time
from sensor_msgs.msg import Range
from visualization_msgs.msg import MarkerArray,Marker

rospy.init_node('flight')

marker_pub = rospy.Publisher("/visualization_markers", MarkerArray, queue_size = 2)

#Создаем MarkerArray для вывода стен
markerArray = MarkerArray()

#Отмечаем стены по координатам начала и конца
def Mark_Wall(x1,y1,x2,y2,id,maxid):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.type = 1
    marker.id = id
    #Задаем размеры стены
    marker.scale.x = abs(x1-x2)+0.01
    marker.scale.y = abs(y1-y2)+0.01
    marker.scale.z = 1.5
    #Задаем цвет стены
    marker.color.r = 1.0*(1-id/maxid)
    marker.color.g = 1.0*(id/maxid)
    marker.color.b = 1.0
    marker.color.a = 1.0
    #Задаем координаты центра стены
    marker.pose.position.x = (x1+x2)/2
    marker.pose.position.y = (y1+y2)/2
    marker.pose.position.z = 0
    #Задаем поворот стены
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    # выводим стены
    print("Wall ",i+1,': ',abs(x1-x2)+abs(x1-x2),'\n',end='')
    return marker
#Массив координат краёв стен
cord=[(1.0,3.0),(3.0,3.0),(3.0,4.0),(4.0,4.0),(4.0,1.0),(5.0,1.0),(5.0,4.0),(6.0,4.0),(6.0,1.0),(7.0,1.0),(7.0,4.0)]

#Добавляем стены в массив маркеров
for i in range(len(cord)-1):
    markerArray.markers.append(Mark_Wall(cord[i][0],cord[i][1],cord[i+1][0],cord[i+1][1],i,len(cord)-1))
    


bridge = CvBridge()



get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
land = rospy.ServiceProxy('land', Trigger)

def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.1):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

def land_wait():
    land()
    while get_telemetry().armed:
        rospy.sleep(0.2)

color_debug = rospy.Publisher("/color_debug", Image, queue_size=1)
land_debug = rospy.Publisher("/land_debug", Image, queue_size=1)
# hsv-диапазоны для посадки на H-зону
land_min = [68, 20, 95]
land_max = [90, 60, 161]
img = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')
x_center, y_center = img.shape[1] / 2, img.shape[0] / 2 # Центр изображения

# Всё для PID-регулятора
oldtime = 0
realtime = 0
# X PID
y_preverr = 1000000000
PX = 0
IX = 0
DX = 0
# Y PID
x_preverr = 1000000000
PY = 0
IY = 0
DY = 0
# Yaw PID
Pyaw = 0
Iyaw = 0
Dyaw = 0
preverryaw = 1000000000
# Z PID
PZ = 0
IZ = 0
DZ = 0
z_preverr = 1000000000
k = 0
# Коэффициенты PID по Y
Pky = 0.0016
Iky = 0.01
Dky = 0.00016
# Коэффициенты PID по X
Pkx = 0.0016
Ikx = 0.01
Dkx = 0.00016
# Коэффициенты PID по Yaw
Pkyaw = 0.0032
Ikyaw = 0.01
Dkyaw = 0.00016
# Коэффициенты PID по Z
Pkz = 0.08
Ikz = 0.02
Dkz = 0.00016

# Для подсчета корректировки
def PIDresult(Pk, Ik, Dk, P, I, D, err, preverr, tim): # Принимает: 3 коэффициента усиления PID, 3 значения PID-составляющих, ошибку, предыдущую ошибку, время между подсчетами
    P = err # Считаем P составляющую
    if tim < 2:
        I = I + err * tim # Считаем I составляющую
    if preverr != 1000000000 and tim < 2:
        D = (err-preverr) / tim # Считаем D составляющую
    out = P * Pk + I * Ik + D * Dk # Складываем составляющие, умножая их на коэффициенты
    return out

def land_image(data):
    global oldtime, x_preverr, y_preverr, z_preverr, h_land
    realtime = time.time()
    land_cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    land_cv_image_copy = land_cv_image.copy() # Копия изображения
    land_hsv = cv2.cvtColor(land_cv_image, cv2.COLOR_BGR2HSV) # Переводим из BGR в HSV
    land_mask = cv2.inRange(land_hsv, np.array(land_min), np.array(land_max)) # Маска
    land_debug.publish(bridge.cv2_to_imgmsg(land_mask, '8UC1')) # Публикуем в топик
    contours, _ = cv2.findContours(land_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Search moments
    contours.sort(key=cv2.minAreaRect)  # Sort moments
    # Если есть посадочная площадка, летим к ней
    if len(contours) > 0:
        cnt = contours[0]
        if cv2.contourArea(cnt) > 50:
            rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
            (x_min, y_min), (w_min, h_min), angle = rect
            height = rospy.wait_for_message('rangefinder/range', Range).range # Текущая высота
            cv2.drawContours(land_cv_image_copy, contours, 0, (180, 105, 255), 3)  # Обводим посадочную площадку
            color_debug.publish(bridge.cv2_to_imgmsg(land_cv_image_copy, 'bgr8')) # Публикуем в топик
            y_err = (x_center - x_min) # Ошибка по y
            x_err = (y_center - y_min) # Ошибка по x
            z_err = h_land - height # Ошибка по z
            Xout = PIDresult(Pkx, Ikx, Dkx, PX, IX, DX, x_err, x_preverr, realtime - oldtime) # Обработанная ошибка по x
            Yout = PIDresult(Pky, Iky, Dky, PY, IY, DY, y_err, y_preverr, realtime - oldtime) # Обработанная ошибка по y
            Zout = PIDresult(Pkz, Ikz, Dkz, PZ, IZ, DZ, z_err, z_preverr, realtime - oldtime) # Обработанная ошибка по z
            x_preverr = x_err # Запоминаем ошибку по x
            y_preverr = y_err # Запоминаем ошибку по y
            z_preverr = z_err # Запоминаем ошибку по z
            set_velocity(vx=Xout, vy=Yout, vz=Zout, frame_id='body', yaw=float('nan'))
            # Запоминаем время для PID-регулятора
            oldtime = time.time()

            
marker_pub.publish(markerArray)
navigate_wait(z=1, frame_id='body', auto_arm=True)
navigate_wait(x=0, y=4, z=1, frame_id='aruco_map')
navigate_wait(x=7, y=4, z=1, frame_id='aruco_map')
navigate_wait(x=7, y=0, z=1, frame_id='aruco_map')
navigate_wait(x=0, y=0, z=1, frame_id='aruco_map')
navigate_wait(x=0.5, y=0.5, z=1, frame_id='aruco_map')
h_land = 0.8
land_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, land_image, queue_size=1)
rospy.sleep(5)
h_land = 0.5
rospy.sleep(3)
land_sub.unregister()
set_position(frame_id="aruco_map")
land_wait()