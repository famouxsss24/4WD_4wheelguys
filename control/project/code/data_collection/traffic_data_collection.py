import time
import os
import sys
import select
import tty
import termios
from PIL import Image
import mycamera

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    camera = mycamera.MyPiCamera(320, 240)
    save_dir = "/home/fourwheel/Desktop/project/code/traffic_data"
    
    # 저장 폴더가 없으면 생성
    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
        #print(f"[{save_dir}] 폴더가 생성되었습니다.")

    print("====================================")
    print("🚦 표지판/객체 데이터 수집기 🚦")
    print("====================================")
    
    # 기존 터미널 설정 백업
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        while camera.isOpened():
            # 1. 객체 이름 입력받기 (cbreak 모드가 아닐 때 정상적인 input 사용 가능)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            print("\n------------------------------------")
            print("수집할 객체를 번호로 선택하세요:")
            print("1. 좌회전 (left)")
            print("2. 우회전 (right)")
            print("3. 직진 (straight)")
            print("4. 20제한 (limit)")
            print("5. 신호등 (traffic)")
            print("6. 경적 (brr)")
            print("7. 스탑 (stop)")
            print("8. 배경 (background)")
            print("q. 수집기 종료")
            
            choice = input("번호 입력: ").strip()
            
            if choice.lower() == 'q':
                print("수집기를 정상 종료합니다.")
                break
                
            object_map = {
                '1': 'left', '2': 'right', '3': 'straight',
                '4': 'limit', '5': 'traffic', '6': 'brr',
                '7': 'stop', '8': 'background'
            }
            
            if choice not in object_map:
                print("잘못된 입력입니다. 1~8 사이의 번호나 'q'를 입력해주세요.")
                continue
                
            obj_name = object_map[choice]
                
            print(f"\n[{obj_name}] 데이터 수집 준비 완료!")
            print(" - 'k' 키를 누르면 수집 시작 (초당 5장)")
            print(" - 'c' 키를 누르면 수집 중지 및 새 객체 입력으로 돌아감")
            print("------------------------------------\n")

            # 덮어쓰기 방지를 위해 기존 파일들의 최대 인덱스 찾기
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(obj_name + "_") and f.endswith(".jpg")]
            img_count = 0
            if existing_files:
                indices = []
                for f in existing_files:
                    try:
                        # 파일명에서 숫자 부분만 추출
                        idx_str = f.replace(obj_name + "_", "").replace(".jpg", "")
                        indices.append(int(idx_str))
                    except ValueError:
                        pass
                if indices:
                    img_count = max(indices) + 1
            
            is_capturing = False
            last_save_time = 0
            save_interval = 1.0 / 3.0  # 초당 3장
            
            # 터미널 키보드 감지 모드(cbreak) 켜기
            tty.setcbreak(sys.stdin.fileno())
            
            # 수집 루프
            while True:
                # 키보드 입력 확인
                if isData():
                    key = sys.stdin.read(1)
                    if key == 'k' and not is_capturing:
                        is_capturing = True
                        sys.stdout.write(f"\n>>> [{obj_name}] 수집 시작! ('c'를 눌러 중지)\n")
                        sys.stdout.flush()
                        last_save_time = time.time() # 시작 시점부터 타이머 리셋
                    elif key == 'c':
                        is_capturing = False
                        sys.stdout.write(f"\n||| [{obj_name}] 수집 중지됨. 현재까지 총 {img_count}장 수집완료.\n")
                        sys.stdout.flush()
                        break # c를 누르면 내부 루프 탈출 후 새로운 객체 이름 입력으로 이동
                
                # 카메라 프레임 읽기
                _, image = camera.read()
                
                # 카메라 물리적 방향 (상하좌우 반전)
                image = image[::-1, ::-1, :]
                
                # 색상 채널 교정: 파이카메라 배열이 BGR 순서로 들어오고 있으므로 RGB로 변환하여 PIL에 전달
                image = image[:, :, ::-1]
                
                if is_capturing:
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        filename = os.path.join(save_dir, f"{obj_name}_{img_count:04d}.jpg")
                        
                        # 💡 mycamera는 정상적인 RGB를 주고, PIL은 RGB를 그대로 받아 자연스러운 색상(사람이 보는 색상)으로 저장합니다.
                        Image.fromarray(image).save(filename)
                        
                        img_count += 1
                        last_save_time = current_time
                        
                        sys.stdout.write(f"\r수집 중... {filename} 저장됨")
                        sys.stdout.flush()
                
                time.sleep(0.01) # CPU 점유율 방지를 위한 미세한 딜레이
                
    except KeyboardInterrupt:
        print("\n프로그램을 강제 종료합니다.")
    finally:
        # 프로그램 종료 시 터미널 설정 무조건 복구
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()
