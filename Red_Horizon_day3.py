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
import requests

bridge = CvBridge()

rospy.init_node('flight')

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



fire_arr = []
hurt_arr = []
# hsv-диапазоны для распознования возгорания, пострадавшего и зоны посадки
fire_min, fire_max = [9, 68, 121], [35, 180, 255]
hurt_min, hurt_max = [90, 75, 0], [255, 255, 255]
land_min, land_max = [68, 20, 95], [90, 60, 161]

fire_detect = rospy.Publisher("/fire_detect", Image, queue_size=1)
hurt_detect = rospy.Publisher("/hurt_detect", Image, queue_size=1)
color_debug = rospy.Publisher("/color_debug", Image, queue_size=1)
land_debug = rospy.Publisher("/land_debug", Image, queue_size=1)

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

def image_callback(data):
    # возгорание
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    fire_cv_image = cv_image.copy()
    fire_cv_image_copy = cv_image.copy()
    fire_hsv = cv2.cvtColor(fire_cv_image, cv2.COLOR_BGR2HSV)  # меняем BGR в HSV
    fire_img = cv2.inRange(fire_hsv, np.array(fire_min), np.array(fire_max))  # цветокор для нахождения возгорания
    contours_fire, _ = cv2.findContours(fire_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # поиск контуров возгорания
    contours_fire.sort(key=cv2.minAreaRect)  # сортировка контуров по площади
    # если есть возговрание
    if len(contours_fire) > 0:
        cnt_fire = contours_fire[0]
        if cv2.contourArea(cnt_fire) > 100:
            rect_fire = cv2.minAreaRect(cnt_fire)  # вписываем прямоугольник для поиска координат возгорания на изображении
            (x_minkvad, y_minkvad), (w_minkvad, h_minkvad), angle = rect_fire
            cv2.drawContours(fire_cv_image_copy, contours_fire, 0, (180, 105, 255), 3) # обводим возгорание
            fire_detect.publish(bridge.cv2_to_imgmsg(fire_cv_image_copy, 'bgr8')) # публикуем обработанное изображение в fire_detect
            # если возгорание близко к центру изображения, определяем его координаты (координаты коптера)
            if abs(y_minkvad - y_center) < 75 and abs(x_minkvad - x_center) < 75:
                telem_im = get_telemetry(frame_id='aruco_map')
                xfire = telem_im.x
                yfire = telem_im.y
                # проверяем, что это не тот же самое возгорание
                if len(fire_arr) > 0:
                    for fire_info in fire_arr:
                        x_dop, y_dop = fire_info[1], fire_info[2]
                        if math.sqrt((x_dop - xfire) ** 2 + (y_dop - yfire) ** 2) < 0.3:
                            break
                    else:
                        # Запрос на сервер
                        r = requests.get('http://65.108.222.51/check_material?x={x:.2f}&y={y:.2f}'.format(x=xfire, y=yfire), auth=('user', 'pass'))
                        r = r.text
                        # Обработка
                        if r.count("{") > 0:
                            ans = "-"
                        else:
                            material = r.split('"')[1]
                            if material == "coal" or material == "textiles" or material == "plastics":
                                clas = "A"
                                dop = [xfire, yfire, material, clas]
                            elif material == "oil" or material == "alcohol" or material == "glycerine":
                                clas = "B"
                                dop = [xfire, yfire, material, clas]
                            fire_arr.append(dop)
    # пострадавший
    hurt_cv_image = cv_image.copy()
    hurt_cv_image_copy = hurt_cv_image.copy()
    hurt_hsv = cv2.cvtColor(hurt_cv_image, cv2.COLOR_BGR2HSV)  # меняем BGR в HSV
    hurt_img = cv2.inRange(hurt_hsv, np.array(hurt_min), np.array(hurt_max))  # цветокор для нахождения пострадавшего
    contours_hurt, _ = cv2.findContours(hurt_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # обводим пострадавшего
    contours_hurt.sort(key=cv2.minAreaRect)  # сортировка контуров по площади
    # Если есть пострадавший
    if len(contours_hurt) > 0:
        cnt_hurt = contours_hurt[0]
        if cv2.contourArea(cnt_hurt) > 100:
            rect_hurt = cv2.minAreaRect(cnt_hurt)  # вписываем прямоугольник для поиска координат пострадавшего на изображении
            (x_minkvad, y_minkvad), (w_minkvad, h_minkvad), angle = rect_hurt
            cv2.drawContours(hurt_cv_image_copy, contours_hurt, 0, (180, 105, 255), 3)  # обводим пострадавшего
            hurt_detect.publish(bridge.cv2_to_imgmsg(hurt_cv_image_copy, 'bgr8'))  # публикуем обработанное изображение в hurt_detect
            # если пострадавший близко к центру изображения, определяем его координаты (координаты коптера)
            if abs(y_minkvad - y_center) < 75 and abs(x_minkvad - x_center) < 75:
                telem_im = get_telemetry(frame_id='aruco_map')
                xhurt = telem_im.x
                yhurt = telem_im.y
                # проверяем, что это не тот же самый пострадавший
                if len(hurt_arr) > 0:
                    for hurt_info in hurt_arr:
                        x_dop, y_dop = hurt_info[1], hurt_info[2]
                        if math.sqrt((x_dop - xhurt) ** 2 + (y_dop - yhurt) ** 2) < 0.3:
                            break
                    else:
                        mn = 999999999
                        count = 1
                        count_mn = -1
                        for fire in fire_arr:
                            x_fire, y_fire = fire[0], fire[1]
                            rast = math.sqrt((xhurt-x_fire) ** 2 + (yhurt-y_fire) ** 2)
                            if rast < mn:
                                mn = rast
                                count_mn = count
                            count += 1
                        dop = [xhurt, yhurt, count_mn]
                        hurt_arr.append(dop)

navigate_wait(z=1, frame_id='body', auto_arm=True) # Взлёт
telem = get_telemetry(frame_id='aruco_map')
xstart, ystart = telem.x, telem.y # Запоминаем координаты зоны взлёта/посадки
points = [[0.0, ystart], [0.0, 0.0], [0.0, 3.0], [1.5, 3.0], [1.5, 4.0], [4.0, 4.0], [4.0, 1.0], [4.0, 2.5], [7.0, 2.5], [7.0, 0.5], [7.0, 3.5], [4.0, 3.5], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0], [xstart, ystart]]

fire_hurt_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback, queue_size=1) # Вкл мониторинг

for point in points:
    x_point, y_point = point[0], point[1]
    navigate_wait(x=x_point, y=y_point, z=0.7, frame_id='body', auto_arm=True)  # Полёт по точкам

fire_hurt_sub.unregister() # Откл мониторинг

# Посадка
h_land = 0.8
land_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, land_image, queue_size=1)
rospy.sleep(5)
h_land = 0.5
rospy.sleep(3)
land_sub.unregister()
set_position(frame_id='aruco_map')
land_wait()

print("Fires: {count}".format(len(fire_arr)))
count = 1
for fire in fire_arr:
    x, y, material, clas = fire[0], fire[1], fire[2], fire[3]
    print("Fire {}: {} {} {} {}".format(count, x, y, material, clas))
    count += 1

print("Injured: {}".format(len(hurt_arr)))
count = 1
for hurt in hurt_arr:
    x, y, fire = hurt[0], hurt[1], hurt[2]
    print("Injured {}: {} {} {}".format(count, x, y, fire))
    count += 1
