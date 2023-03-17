# Код команды Red Horizon

Для решения данной задачи мы использовали:
- Для полёта - функцию navigate_wait(), которая принимает координаты и ожидает прилёта в заданную точку.
- Для ожидания посадки и дизарма коптера - функцию land_wait().
- Для распознавания стен, возгораний и пострадавших - библиотеку cv2.
- Для центрирования мы использовали ПИД регулятор.

Алгоритм обработки изображения (image_callback()):
Определяем цветовые диапазоны возгораний и пострадавших.
Преобразуем изображение, полученное с камеры, в HSV-изображение.
Применяем к нему цветокоррекцию с заранее подобранными цветовыми диапазонами.
С помощью cv2.findContours() и cv2.minAreaRect ищем контуры возгораний и пострадавших по наибольшей площади.
Если возгорание или пострадавший находится приблизительно в центре изображения, запоминаем текущие координаты коптера.
Возгорание: Если раннее не было распознано данное возгорание и OpenAPI корректно вернул материал, то записываем информацию о возгорании.
Пострадавший: находим ближайшее возгорание и записываем информацию о пострадавшем.
В отладочные Image топики публикуем обработанные изображения.

Алгоритм центрирования на зоной посадки H (land_image()):
Считаем ошибки (отклонение от зоны посадки) по X, Y для того чтобы удерживать коптер над зоной посадки, спускаясь на высоту 0.8м -> 0.5м, всё это работает при помощи ПИД регулятора.

После центрирования над зоной посадки используем land_wait().

***

# Инженерное устройство

#### День 1
Мы обдумывали варианты и возможности решения поставленной задачи, рассматривая различные типы конструкций и их недостатки. Так, сразу была отклонена идея использования шестерней. Позже выяснилось, что и не зря, ведь их пришлось бы печатать на 3D-принтере, что у нас и так очень плохо получалось. 
Также рассматривалась конструкция, предусматривающая вертикальный сброс капсулы пожаротушения, используя их вертикальное расположения и люки сброса, но здесь вставал очень резонный вопрос, получится ли сервоприводу быстро закрывать люк, пока следующая капсула его не заблокировала. 
В итоге всё свелось к использованию конструкции барабанного типа, в которой капсулы располагаются по окружности, а сервопривод, вращающий диск с отверстием, в центре. Но следовало более точно продумать расположение капсул, чем мы занимались во второй день.

>Устройство:
![](https://github.com/Daniil10001/Red_Horizon_final/blob/e4337201a7eb08e5ef00645e82fb35d901283677/construction_photo.jpg)
#### День 2
Первым делом,  устройство было создано в качестве 3D-модели. Было решено сделать 2 этажа: на нижнем находится 4 капсулы пожаротушения типа А и 2 капсулы типа Б, а на втором -- ещё 2 капсулы типа Б. Также была написана документация, а некоторые части мы поставили на печать.
>Документация на Google Drive: [ссылка][s1]

#### День 3
Предстаяло решать множество проблем с печатью, так как принтеры постоянно выходили из строя, и детали оставались либо недопечатанными, либо вообще не печатались. Из-за этого приходилось использовать довольно изощрённые методы сборки: например, мы взяли недопечатанную часть детали и другую недопечатанную часть этой же детали, и склеели их термоклеем.
Наконец, была произведена полная сборка, после которой мы начали писать программный код для устройства. Но основная программа была написана уже на следующий день.

#### День 4
Переписав несколько раз код, в конце концов мы пришли лишь к ручному вводу необходимых комманд в Serial monitor Ардуино.

> В итоге код получился таким:

    #include <Servo.h>  
    Servo servo;  
    int val;
    int ch = 0;
    int fir = 0;
    int sec = 0;
    int thr = 0;
    void setup() {
      Serial.begin(9600);
      servo.attach(10); 
    }
    
    void loop() {
      servo.write(90);
      delay(1000);
      if (Serial.available()) {
        val = Serial.parseInt();
        if (val == 1){
          Serial.println("+120");
          servo.write(180); //ставим вал под 0
          delay(125);
        }
        if (val == 2){
          Serial.println("-120");
          servo.write(0); //ставим вал под 0
          delay(125);
        }
        if (val == 3){
          Serial.println("+85");
          servo.write(180); //ставим вал под 0
          delay(85);
        }
        if (val == 4){
          Serial.println("-85");
          servo.write(0); //ставим вал под 0
          delay(85);
        }
        if (val == 5){
          Serial.println("+25");
          servo.write(180); //ставим вал под 0
          delay(25);
        }
        if (val == 6){
          Serial.println("-25");
          servo.write(0); //ставим вал под 0
          delay(25);
        }
      }
    }

Устройство не смогло автоматически справляться с поставленной задачей, хотя за конструкционное решение мы смогли получить вполне неплохие баллы, если не учитывать то, что нам их сократили в 2 раза за сдачу на следующий день.

[s1]: <https://drive.google.com/drive/folders/1aDbrRLrWBJuIHzo14rJD_ask39b8q_GX?usp=sharing>
