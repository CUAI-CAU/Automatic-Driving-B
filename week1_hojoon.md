## Week1 Summary

by 자율주행 스터디 B조 음호준

### CV에 대하여

#### CV란?

컴퓨터를 이용하여 정지 영상 또는 동영상으로부터 의미있는 정보를 추출하는 방법을 연구하는 학문.

---

#### CV의 응용

- 영상의 화질 개선
- 객체 검출과 영상 분할
- 객체 인식
- 머신 비전 (공장의 자동화 등)
- 인공지능 서비스 (인공지능 로봇 등)

---

### CH 1

#### - 이미지의 구성

    컬러(RGB), 그레이 스케일
    각 화소 값은 1byte.

    3차원 값으로 구성. (Width, height, color)

    png는 알파채널(투명도)지원, gif는 잘 사용하지 않음.

    cv와 cv2 두 라이브러리중 최신의 cv2를 사용함.
    주로 http://docs.opencv.org/master/ 에서 api 명세를 확인할 수 있음.
    파이썬 버전과 openCV버전이 호환이 되지 않을 수 있으므로 확인하여 설치해야함

#### - 이미지를 창에 띄우는 방법.

```python
# 이미지 파일을 불러옵니다.
img = cv2.imread('cat.bmp')
# flags=cv2.IMREAD_GRAYSCALE
if img is None:
	print('img is none')
	sys.exit()
else:
	# namedWindow에 cv2.normalresize를 인자로 넣어주면 이미지창을
    # resize 할 수 있다.
	cv2.namedWindow('image')
	cv2.imshow('image', img)
	# 흑백 사진으로 저장
	# imshow 와 waitkey를 같이 써야 이미지를 볼 수 있다.
    # 불러온 이미지를 새로운 이미지파일로 저장할 수 있다.
	cv2.imwrite('catgray.jpg', img)
	cv2.waitKey()
	# cv2.destroyWindow는 window의 이름을 받아 창을 닫는다.
	cv2.destroyAllWindows()
```

---

#### - matplotlib을 이용해서 이미지를 출력하기

imread를 이용해서 불러온 이미지의 화소값은 BGR이기 때문에 이를 RGB로 변환해야함.

**cv2.cvtColor()** 사용

**_plt.show()_** 를 사용해서 출력.

**_subplot()_** 을 이용해서 하나의 창에 여러 개의 이미지 출력도 가능.

슬라이드쇼 출력 예시

```python
# 이미지 파일을 모두 img_files 리스트에 추가
# 경로가 절대경로이므로 아래와 같이 설정.
img_files = glob.glob('./fastCampusOpenCv/ch01/images/*.jpg')

if not img_files:
    print("There are no jpg files in 'images' folder")
    sys.exit()

# 전체 화면으로 'image' 창 생성
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 무한 루프
cnt = len(img_files)
idx = 0

while True:
    img = cv2.imread(img_files[idx])

    if img is None:
        print('Image load failed!')
        break

    cv2.imshow('image', img)
    if cv2.waitKey(1000) >= 0:
        break

    idx += 1
    if idx >= cnt:
        idx = 0

cv2.destroyAllWindows()
```

### CH 2

#### - openCV 데이터

openCV 영상 데이터는 numpy의 ndarray이다.
3차원 데이터이며, 데이터 타입은 uint8이다.

_for loop_ 를 통해 픽셀값을 변경하는 것은
매우 느리므로 사용하지 말것. 다음과 같이 사용하자.

```python
img[:,:] = 255 # 그레이 스케일
img[:,:] = (0, 0, 255) # 트루 컬러
```

넘파이의 zeros, ones, empty, full을 이용해서
임의의 새로운 영상을 만드는 것도 가능.

불러온 영상을 완전히 복사하기 위해서는 copy()를
사용해야함.  
ndarray를 슬라이싱하는 것으로 영상의 일부만 변경, 추출도 가능.

#### - Region of Interest

마스크 연산을 통해 원하는 이미지 영역만 다른 이미지로 추출, 합성 가능.  
넘파이의 불리언 인덱싱 응용 가능.

```python
dst[mask > 0] = src[mask > 0]
```

#### - 그리기 함수들

```python
cv2.line()
cv2.rectangle()
cv2.circle()
cv2.polylines()
cv2.putText()
```

#### - 카메라와 동영상 처리

#### cv2.VideoCapture() 이용

```python
cv2.VideoCapture.isOpened()
# 비디오 캡쳐 준비 확인
cv2.VideoCapture.read()
# 비디오 프레임 받아오기
cv2.VideoCapture.get()
# 비디오의 속성(가로, 세로, 프레임수, 현재 위치, 노출값 등) 조회

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 카메라 프레임 크기 출력
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 카메라 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    inversed = ~frame  # 반전

    cv2.imshow('frame', frame)
    cv2.imshow('inversed', inversed)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

_cv2.VideoWriter_ 를 이용해 동영상 저장가능.  
영상의 코덱, 압축 방식, 픽셀 포맷등을 Fourcc값으로 지정가능.

#### - 키보드 이벤트

```python
cv2.waitKey()를 이용
입력 받은 키값의 ASCII code 값을 리턴.
리턴 값은 ord('키 입력값')과 동일

```

#### - 마우스 이벤트

cv2.setMouseCallback를 이용

- 마우스 이벤트 처리를 수행할 창이름,
- 이벤트에 대한 콜백함수,
- 콜백함수에 전달할 데이터

를 파라미터로 받음.

#### - 트랙바

트랙바를 이용하여 openCV 윈도우에서 사용자가 임의로 값을 그래픽 인터페이스를 이용해 조정가능.

#### - 연산시간 측정

cv2.TickMeter()를 이용하여 측정가능.

```python
timer = cv2.TickMeter()
timer.start()
~~~
timer.stop()

print(timer.getTimeMilli)
```

### CH 3

#### - 화소 처리

입력 영상의 특정 좌표의 픽셀 값을 변경하여 출력 영상의 해당 좌표에 변경된 픽셀 값으로 변환.

#### - 밝기 조절

_cv2.add()_ 이용
_numpy.clip()_ 을 이용하는 방법도 있음.

#### - 영상의 산술 연산

- 덧셈연산
  _cv2.add()_ 를 이용해 두 이미지 소스의 픽셀을 합성.  
  덧셈 결과가 255보다 크면 255로 설정.

- 가중치 합(weighted sum)  
  두 픽셀값에 각각 가중치를 주어 합을 계산. (가중치는 합이 1이 되도록 설정)  
  _cv2.addWeighted()_ 를 이용.

- 뺄셈연산
  _cv2.subtract()_ 이용.  
  _cv2.add()_ 의 반대.

- 차이연산
  _cv2.absdiff()_ 이용.  
  뺄셈 연산 후 그 차이의 절댓값의 화소값을 결과값으로 이용.

- 비트연산
  각 픽셀값을 이진수로 변환한 후, 비트연산을 수행.

```python
cv2.bitwise_and()
cv2.bitwise_or()
cv2.bitwise_xor()
cv2.bitwise_not()
```
