# 4WD_4wheelguys
광운대학교 임베디드인공지능시스템최적화 Project, "사륜구동" 팀입니다.

표지판·신호등을 인식하며 스스로 달리는 자율주행 RC카. 인식 모델부터 엣지 최적화·주행 제어까지 통합해 **라즈베리파이5에서 실시간 구동**을 달성했습니다.

## 🔧 프로젝트 요약
표지판·신호등 인식 기반 자율주행 RC카를 4인 팀으로 개발했습니다.

- **인식 모델**: YOLOv11n 기반 표지판/신호등 인식
- **엣지 최적화**: ONNX→TFLite INT8 양자화로 라즈베리파이5 실시간 구동 확보 (75%↓, mAP 0.95, 30FPS)
- **주행 제어**: Conditional Imitation Learning 기반 조향 모델과 인식 결과 결합

## 🗺️ 아키텍처 & Flow
```
카메라 → YOLOv11n (표지판·신호등 인식)
      → ONNX → TFLite INT8 양자화 (엣지 경량화 · 75%↓)
      → 인식 결과 + CIL(Conditional Imitation Learning) 조향 모델
      → 주행 제어 (라즈베리파이5 · 실시간 30FPS)
```
정확한 모델도 임베디드 보드에서 실시간으로 돌지 못하면 자율주행에 쓸 수 없다. "정확한가"를 넘어 **"제한된 하드웨어에서 실시간으로 이어지는가"**를 함께 푼 것이 핵심이다.

## 🏆 성과
포스터·실무보고서 부문 우수상 (광운대학교 매치업 심화과정 시상식)

<img width="3401" height="4535" alt="임베디드인공지능시스템최적화_포스터 (900 x 1200 mm) (900 x 1200 mm) pdf" src="https://github.com/user-attachments/assets/a366bda5-b642-41fe-815c-f9be63182590" />

<br>

## 🎬 데모 영상

라즈베리파이 기반 자율주행 RC카의 트랙 주행 데모입니다. 표지판, 신호등과 같은 외부 환경요소를 잘 인식해 미션을 수행하며 안정적으로 주행하는 모습입니다.
아래 영상을 클릭하면 재생됩니다.


https://github.com/user-attachments/assets/5201b997-aafb-4de9-93ed-3cc123f96539


## 📸 프로젝트 갤러리

<table>
  <tr>
    <td width="33%" align="center"><img src="media/IMG_1988.jpg" width="100%"><br><sub>설계 · 학습 전략 회의</sub></td>
    <td width="33%" align="center"><img src="media/IMG_2408.jpg" width="100%"><br><sub>자율주행 트랙 주행</sub></td>
    <td width="33%" align="center"><img src="media/IMG_2456.jpg" width="100%"><br><sub>포스터 발표 심사</sub></td>
  </tr>
  <tr>
    <td width="33%" align="center"><img src="media/IMG_2434.jpg" width="100%"><br><sub>팀 사륜구동</sub></td>
    <td width="33%" align="center"><img src="media/IMG_2452.jpg" width="100%"><br><sub>우수상 수상 (포스터 · 실무보고서 부문)</sub></td>
    <td width="33%" align="center"><img src="media/IMG_2425.jpg" width="100%"><br><sub>매치업 심화과정 시상식</sub></td>
  </tr>
</table>
