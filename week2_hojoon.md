## Week1 Summary

by 자율주행 스터디 B조 음호준
api위주로 정리.

### CH3(컬려영상과 색공간) 부터

openCV에서는 RGB -> BGR

- .split(m, mv=None) -> dst  
  다채널 컬러 영상을 각 채널 별로 분리.

- .merge(mv, dst=None) -> dst  
  채널을 결합.

RGB 채널을 HSV, YCrCb, Grayscale 등의 다른 색공간으로 변환하여 처리 가능.

> HSV 색 공간  
> • Hue: 색상,색의종류  
> • Saturation: 채도, 색의 탁하고 선명한 정도 명도  
> • Value: 빛의밝기

> YCrCb 색 공간  
> • PAL,NTSC, SECAM등의컬러비디오표준에사용되는색공간  
> • 영상의밝기정보와색상정보를따로분리하여부호화(흑백TV호환)  
> • Y: 밝기정보(luma)  
> • Cr, Cb: 색차(chroma)

- .cvtColor(src, code, dst=None, dstCn=None) -> dst  
  code <- 색변환 코드

### 히스토그램 분석

영상의 픽셀값 분포를 그래프로 표현.  
정규화를 통해 각 픽셀이 나타날 확률의 분포를 확인할 수 있음.

- .calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None) -> hist

> • images: 입력 영상 리스트  
> • channels: 히스토그램을 구할 채널을 나타내는 리스트  
> • mask: 마스크 영상. 입력 영상 전체에서 히스토그램을 구하려면 None 지정.  
> • histSize: 히스토그램 각 차원의 크기(빈(bin)의 개수)를 나타내는 리스트  
> • ranges: 히스토그램 각 차원의 최솟값과 최댓값으로 구성된 리스트  
> • hist: 계산된 히스토그램 (numpy.ndarray)  
> • accumulate: 기존의 hist 히스토그램에 누적하려면 True, 새로 만들려면 False.

### 명암비 조절

영상의 그레이스케일 히스토그램 스트레칭을 통해 영상의 전구간에서 히스토그램이 나타나도록 하여 명암비 조절 가능.

- .normalize(src, dst, alpha=None, beta=None, norm_type=None, dtype=None, mask=None) -> dst

히스토그램 평활화를 통해 그레이스케일이 영상 전체 구간에서 균등하게 나타나도록하여 명암비 향상가능.

- .equalizeHist(src, dst=None) -> dst  
  src: 그레이 스케일 영상.

### 특정 색상 영역 추출

특정 범위 안에 있는 행렬 원소를 검출하는 방법.

- .inRange(src, lowerb, upperb, dst=None) -> dst

하한과 상한의 행렬 혹은 스칼라값을 입력하여 그 영역의 행렬 원소를 검출

### 히스토그램 역투영

> 영상의 각 픽셀이 주어진 히스토그램 모델에 얼마나 일치하는 지를 검사하는 방법.  
> 임의의 색상 영역을 검출할 때 효과적임.

1. 기준 영상으로부터 살색에 대한 컬러 히스토그램을 미리 계산하고,
2. 입력 영상에서 미리 구한 살색 히스토그램에 부합하는 픽셀을 선별한다.

- .calcBackProject(images, channels, hist, ranges, scale, dst=None) -> dst  
  • images: 입력영상리스트  
  • channels: 역투영계산에사용할채널번호리스트  
  • hist: 입력히스토그램(numpy.ndarray)  
  • ranges: 히스토그램각차원의최솟값과최댓값으로구성된리스트  
  • scale: 출력역투영행렬에추가적으로곱할값  
  • dst: 출력역투영영상.입력영상과동일크기,cv2.CV_8U.

### CH4 필터링

필터링은 영상에서 필요한 정보만을 통과시키고 원하지 않는 저오는 걸러내는 작업을 말한다.

- 주파수 공간 필터링
- 공간적 필터링  
  영상의 픽셀 값을 직접 이용.  
  주로 마스크 연산을 이용.(영상 부드럽게 만들기, 영상 날카롭게 만들기, 에지 검출, 잡음 제거)

공간적 필터링을 위한 가장자리 픽셀 확장 방법에는 여러가지가 있다.(상수복사, 확장, 반사 등)

- .filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None) -> dst  
  • src: 입력 영상  
  • ddepth: 출력 영상 데이터 타입. (e.g) cv2.CV_8U, cv2.CV_32F, cv2.CV_64F -1을 지정하면 src와 같은 타입의 dst 영상을 생성  
  • kernel: 필터 마스크 행렬. 실수형.  
  • anchor: 고정점 위치. (-1, -1)이면 필터 중앙을 고정점으로 사용  
  • delta: 추가적으로 더할 값  
  • borderType: 가장자리 픽셀 확장 방식

#### 평균값 필터링을 이용해 영상을 부드럽게 만들 수 있다.

- .blur(src, ksize, dst=None, anchor=None, borderType=None) -> dst

ksize: 평균값필터크기.(width,height)형태의튜플.

평균값 필터의 단점을 보완하기 위해 가우시안 필터를 사용하기도 한다.

- .GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None) -> dst

가우시안 필터를 사용해 언샤프 마스크 필터링을 할 수 있다.

미디언 필터를 이용해 영상의 노이즈를 감소시킬 수 있다.

양방향 필터를 이용해서 에지를 보전하며 노이즈를 제거할 수 있다.

- .bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None) -> dst

### CH 5 영상의 기하학적 변환

영상의 기하학적으로 변환하여 다른 모양을 만든다.

#### 어파인 함수

- .warpAffine(src, M, dsize, dst=None, flags=None,
  borderMode=None, borderValue=None) -> dst

M: 2x3 실수형 변환행렬을 받아 변환한다.

#### 영상의 크기 변환

- .resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None) -> dst

#### 영상의 대칭 변환

- .flip(src, flipCode, dst=None) -> dst  
  • src: 입력영상  
  • flipCode: 대칭방향지정(1,0,-1: 좌우, 상하, 좌우&상하)

#### 이미지 피라미드

하나의 영상을 여러가지 해상도로 변환하여 세트를 구성한것. 보통 가우시안 블러링, 다운 샘플링 형태로 변환하여 구성한다.

- .pyrDown(src, dst=None, dstsize=None, borderType=None) -> dst

- .pyrUp(src, dst=None, dstsize=None, borderType=None) -> dst

#### 회전변환

- .getRotationMatrix2D(center, angle, scale) -> retval
  retval은 2x3 어파인 변환 행렬.

> 어파인 변환 행렬 구하기

- .getAffineTransform(src, dst) -> retval
  retval은 2x3 투시 변환 행렬
- .getPerspectiveTransform(src, dst, solveMethod=None) -> retval
  retval은 3x3 투시 변환 행렬

> 투시 변환 함수

- .warpPerspective(src, M, dsize, dst=None, flags=None,
  borderMode=None, borderValue=None) -> dst

#### 리매핑

영상의 특정 위치의 픽셀을 다른 위치로 재배치 한다.

- .remap(src, map1, map2, interpolation, dst=None, borderMode=None,
  borderValue=None) -> dst

### CH 6 영상의 특징 추출

에지 검출이란 영상에서 픽셀의 밝기 값이 급격하게 변하는 부분을 검출하는 것을 말한다.

#### 소벨 필터 이용

- .Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None,
  delta=None, borderType=None) -> dst

#### 샤르 필터 이용

- .Scharr(src, ddepth, dx, dy, dst=None, scale=None, delta=None,
  borderType=None) -> dst

#### 그래디언트

영상의 픽셀의 스칼라 값을 벡터로 변환.

벡터의 크기 계산

- .magnitude(x, y, magnitude=None) -> magnitude(2d 벡터의 크기 행렬)

- .phase(x, y, angle=None, angleInDegrees=None) -> angle(2d 벡터의 크기 행렬)

#### 캐니 에지 검출

J. 캐니가 고안한 정확한 에지를 검출하는 프로세스

- .Canny(image, threshold1, threshold2, edges=None, apertureSize=None,
  L2gradient=None) -> edges

#### 허프 변환 직선 검출

2차원 영상 좌표에서 직선의 방정식을 파라미터 공간으로 변환하여 직선을 찾는 알고리즘

허프 변환 선분 검출

- .HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None) -> lines

확률적 허프 변환에 의한 선분 검출

- .HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) -> lines

허프 변환 원 검출

- .HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None) -> circles

### CH 7 영상의 이진화

영상의 픽셀 값을 기준에 따라 0 또는 1로 변환

- .threshold(src, thresh, maxval, type, dst=None) -> retval, dst
  retval: 사용된 임계값
  dst: 출력 영상

#### 자동 이진화

Otsu 이진화: 입력 영상이 배경(background)과 객체(object)두개로 구성되어있다고 가정,  
임의의임계값T에의해나눠지는두픽셀분포그룹의분산이최소가되는T를선택

threshold 함수의 thresh parameter의 값을 otsu로 설정.

#### 지역 이진화

균일하지 않은 조명 환경에서 촬영된 영상을 불균일한 조명 성분을 보정한 후 이진화

#### 적응형 이진화

- .adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C,dst=None) -> dst

#### 모폴로지

영상을 형태학적으로 다루는 기법.  
영상 처리 시스템에서 전처리 또는 후처리 형태로 주로 사용됨

침식 -> 객체 외곽을 깎아내는 연산
팽창 -> 객체 외곽을 확대시키는 연산

- .erode(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None) -> dst

- .dilate(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None) -> dst

모폴로지 구조 요소(kernel)은 다음으로 얻어진다.

- .getStructuringElement(shape, ksize, anchor=None) -> retval

범용 모폴로지 연산 함수(침식과 팽창, 열기와 닫기 범용)

열기 연산 = 침식 -> 팽창
닫기 연산 = 팽창 -> 침식

#### 레이블링

레이블링은 동일 객체에 속한 모든 픽셀에 고유한 번호를 매기는 작업이다.

- .connectedComponents(image, labels=None, connectivity=None,
  ltype=None) -> retval, labels

객체 정보를 함께 반환하는 레이블링 함수

- .connectedComponentsWithStats(image, labels=None, stats=None, centroids=None, connectivity=None, ltype=None) -> retval, labels, stats, centroids

#### 외곽선 검출

객체의 외곽선 좌표를 모두 추출하는 작업

- .findContours(image, mode, method, contours=None, hierarchy=None, offset=None) -> contours, hierarchy

검출된 외곽선을 그리는 것도 가능하다.

- .drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
  -> image

외곽선의 길이

- .arcLength(curve, closed) -> retval

외곽선의 면적

- .contourArea(contour, oriented=None) -> retval

바운딩 박스(외곽선을 외접하여 둘러싸는 가장 작은 사각형)

- .boundingRect(array) -> retval

바운딩 써클(외곽선을 외접하여 둘러싸는 가장 작은 원)

- .minEnclosingCircle(points) -> center, radius

외곽선을 근사시킨 곡선

- .approxPolyDP(curve, epsilon, closed, approxCurve=None) -> approxCurve

Convex 검사

- .isContourConvex(contour) -> retval

### CH 8 영상 분할과 객체 검출

#### 그랩 컷(GrabCut)

그래프 컷 기반의 영역 분할 알고리즘.  
영상의 픽셀을 그래프 정점으로 간주하고,  
픽셀들을 2개의 그룹으로 나누는 최적의 컷을 찾는 방식

- .grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None) -> mask(입출력 마스크), bgdModel(임시 배경 모델 행렬), fgdModel(임시 전경 모델 행렬)

#### 모멘트 기반 객체 검출

모멘트란 영상의 형태를 표현하는 일련의 실수값이다.  
 특정 함수 집합과의 상관 관계 형태로 계산된다.

- .matchShapes(contour1, contour2, method, parameter) -> retval(두 외곽선 또는 그레이 스케일 영상 사이의 거리)

#### 템플릿 매칭

특정 템플릿 영상과 일치하는 부분을 찾는 기법.

- .matchTemplate(image, templ, method, result=None, mask=None) -> result(비교 결과 행렬)

#### 케스케이드 분류기

##### viola-jones 얼굴 검출기

Positive영상(얼굴영상)과negative영상(얼굴아닌영상)을훈련하여빠르고 정확하게 얼굴 영역을 검출  
• 기존방법과의차별점  
▪ 유사 하르(Haar-like) 특징을 사용  
▪ AdaBoost에 기반한 강한 분류 성능  
▪ 캐스케이드(cascade) 방식을 통한 빠른 동작 속도  
• 기존얼굴검출방법보다약15배빠르게동작

- .CascadeClassifier()를 이용해
  객체를 생성

- .detectMultiScale(image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None) -> result(검출된 객체의 사각형 정보)

객체의 멀티스케일 함수를 사용해 검출

#### HOG (Histogram of Oriented Gradients)

영상의 지역적 그래디언트 방향 정보를 특징 벡터로 사용하여 보행자를 검출하는 알고리즘

- .HOGDescriptor() 객체 생성

HOG 객체의 .setSVMDetector(svmdetector) -> None 함수를 이용해 SVM 분류기 계수 등록

HOG 객체의 .detectMultiScale(img, hitThreshold=None, winStride=None, padding=None, scale=None, finalThreshold=None,
useMeanshiftGrouping=None) -> foundLocations(검출될 사각형 영역 정보), foundWeights(검출된 사각형 영역에 대한 신뢰도) 함수를 이용해 검출
