import os
import smbus
import time
import matplotlib
import argparse
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import csv
import threading
from datetime import datetime
from ArmIK.ArmMoveIK import *
import pickle
import numpy as np
from scipy.fft import fft, fftfreq
import Board  # 新增的导入，根据你的代码推测可能需要

# 解析命令行参数，修改默认路径为绝对路径（注意：绝对路径应以 "/" 开头）
parser = argparse.ArgumentParser(description='葡萄成熟度检测程序')
parser.add_argument('--model', default='/home/ubuntu/svm_model.pkl', help='模型文件路径')
parser.add_argument('--weights', default='/home/ubuntu/weights.pkl', help='权重文件路径')
parser.add_argument('--max_fft_length', default='/home/ubuntu/max_fft_length.pkl', help='最大FFT长度文件路径')
args = parser.parse_args()

# ---- 硬件初始化 ----
bus = smbus.SMBus(1)
address = 0x48
bus.write_i2c_block_data(address, 0x01, [0x40, 0x83])
time.sleep(0.1)

# ---- 加载模型、权重和最大FFT长度 ----
model_filename = args.model
if not os.path.exists(model_filename):
    print(f"错误: 模型文件 {model_filename} 不存在，请检查路径。")
    exit()
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

weights_filename = args.weights
if not os.path.exists(weights_filename):
    print(f"错误: 权重文件 {weights_filename} 不存在，请检查路径。")
    exit()
with open(weights_filename, 'rb') as f:
    weights = pickle.load(f)

max_fft_length_filename = args.max_fft_length
if os.path.exists(max_fft_length_filename):
    with open(max_fft_length_filename, 'rb') as f:
        max_fft_length = pickle.load(f)
else:
    print(f"错误: 未找到 max_fft_length 文件，请确保你已经运行过训练代码并保存了该文件。")
    exit()

# ---- 全局变量与锁 ----
raw_x_data, raw_y_data = [], []  # 原始数据
smooth_x_data, smooth_y_data = [], []  # 平滑后的数据
data_lock = threading.Lock()
csv_lock = threading.Lock()
recording = False
recording_start_time = 0
current_csv_file = None
current_csv_writer = None
maturity_text = None
arm_running = False  # 记录机械臂运行状态
arm_finished_event = threading.Event()  # 用于通知机械臂运行结束
prediction_result = None  # 新增：用于存储预测结果

# ---- 绘图初始化 ----
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', linewidth=2)
ax.set_ylim(0, 30)
ax.set_xlim(0, 30)
ax.grid(True)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Force (N)", fontsize=12)
fig.canvas.manager.set_window_title('Real-time Force Monitoring')

# 添加记录按钮
ax_button = plt.axes([0.81, 0.01, 0.15, 0.075])
record_button = Button(ax_button, 'Run', color='lightgoldenrodyellow')

# 添加成熟度显示文本，初始为空
maturity_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12, color='black')

# ---- 电压转换函数 ----
def voltage_to_force(v):
    return max(0, (-0.417 * (v ** 2) - 0.949 * v + 2.517) * 9.81)

# ---- 数据平滑函数 ----
def moving_average(data, window_size=5):
    """移动平均滤波"""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        smoothed.append(sum(window) / window_size)
    return smoothed

def exponential_moving_average(data, alpha=0.3):
    """指数加权移动平均（EMA）"""
    if not data:
        return []
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
    return smoothed

# ---- 数据提取和统计计算函数 ----
def extract_force_data_and_calculate_stats(force_data):
    force_data = np.array(force_data)
    force_data = force_data[force_data >= 0]
    if len(force_data) == 0:
        return None

    mean_force = np.mean(force_data)
    std_force = np.std(force_data)
    diff_force = np.diff(force_data)
    rising_rate = np.mean(diff_force[diff_force > 0]) if len(diff_force) > 0 and np.sum(diff_force > 0) > 0 else 0
    peak_force = np.max(force_data)

    N = len(force_data)
    T = 0.1
    yf = fft(force_data)
    xf = fftfreq(N, T)[:N // 2]
    yf_abs = 2.0 / N * np.abs(yf[:N // 2])

    return mean_force, std_force, rising_rate, peak_force, yf_abs

# ---- 特征准备函数 ----
def prepare_features(mean_force, std_force, rising_rate, peak_force, fft_data, max_fft_length):
    if max_fft_length < len(fft_data):
        padded_fft_data = fft_data[:max_fft_length]
    else:
        padded_fft_data = np.pad(fft_data, (0, max_fft_length - len(fft_data)), mode='constant')

    features = [mean_force, std_force, rising_rate, peak_force]
    features.extend(padded_fft_data)
    features = np.array(features)
    return features

# ---- 葡萄类型预测函数 ----
def predict_grape_type(features, model, weights):
    if len(features) != len(weights):
        print("特征长度与权重长度不匹配")
        return None
    weighted_features = features * weights
    weighted_features = weighted_features.reshape(1, -1)
    prediction = model.predict(weighted_features)
    return 0 if prediction[0] == 0 else 1

# ---- 数据采集线程 ----
def collect_data():
    global recording_start_time
    while True:
        try:
            # 读取并转换数据
            data = bus.read_i2c_block_data(address, 0x00, 2)
            raw = (data[0] << 8) | data[1]
            voltage = raw * 4.096 / 32767
            force = voltage_to_force(voltage)
            current_time = time.time()

            with data_lock:
                if recording:
                    elapsed = current_time - recording_start_time
                    # 存入原始数据
                    raw_x_data.append(elapsed)
                    raw_y_data.append(force)

                    # 平滑处理
                    smooth_x = moving_average(raw_x_data, window_size=5)
                    smooth_y = moving_average(raw_y_data, window_size=5)

                    # 控制最大数据点数
                    max_points = 1500
                    if len(smooth_x) > max_points:
                        smooth_x = smooth_x[-max_points:]
                        smooth_y = smooth_y[-max_points:]

                    # 更新全局的平滑数据
                    smooth_x_data.clear()
                    smooth_y_data.clear()
                    smooth_x_data.extend(smooth_x)
                    smooth_y_data.extend(smooth_y)

                    # 写入CSV
                    with csv_lock:
                        if current_csv_writer:
                            current_csv_writer.writerow([
                                f"{elapsed:.3f}",
                                f"{voltage:.4f}",
                                f"{force:.3f}"
                            ])
                            current_csv_file.flush()

            time.sleep(0.02)
        except Exception as e:
            print(f"采集错误: {str(e)}")
            time.sleep(1)

# ---- 动画更新函数 ----
def animate(frame):
    global prediction_result
    with data_lock:
        if not smooth_x_data or not smooth_y_data:
            return line, maturity_text

        current_time = smooth_x_data[-1] if smooth_x_data else 0
        time_window = 30
        ax.set_xlim(
            left=max(0, current_time - time_window),
            right=max(time_window, current_time)
        )

        line.set_data(smooth_x_data, smooth_y_data)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

        # 更新成熟度文本
        if prediction_result is not None:
            if prediction_result == 0:
                maturity_text.set_text('RIPE')
                maturity_text.set_color('green')
            elif prediction_result == 1:
                maturity_text.set_text('ROTTEN')
                maturity_text.set_color('red')
            prediction_result = None  # 重置预测结果

    return line, maturity_text

# ---- 机械臂控制函数 ----
def arm_operation():
    global arm_running
    print("启动机械臂操作...")
    arm_running = True
    maturity_text.set_text('Processing')
    maturity_text.set_color('blue')
    try:
        AK = ArmIK()
        actions = [
            (1, 200, 500), (3, 60, 1000), (4, 650, 1000),
            (5, 350, 1000), (2, 900, 1000), (6, 500, 1000),
            (1, 500, 1500), (1, 380, 1000), (1, 500, 1000),
            (1, 200, 800)
        ]

        for servo_id, pulse, duration in actions:
            setBusServoPulse(servo_id, pulse, duration)
            time.sleep(max(duration / 1000, 0.3))

    except Exception as e:
        print(f"机械臂错误: {str(e)}")
    finally:
        arm_running = False
        arm_finished_event.set()  # 通知机械臂运行结束
        print("机械臂操作完成")

# ---- 成熟度预测函数 ----
def predict_maturity():
    global prediction_result
    print("开始等待机械臂操作完成...")
    arm_finished_event.wait()  # 等待机械臂运行结束
    print("机械臂操作已完成，开始进行成熟度预测...")
    with data_lock:
        if len(smooth_y_data) >= 5:
            print("数据长度足够，开始提取和计算统计数据...")
            result = extract_force_data_and_calculate_stats(smooth_y_data)
            if result is not None:
                print("数据提取和统计计算成功，开始准备特征...")
                mean_force, std_force, rising_rate, peak_force, fft_data = result
                features = prepare_features(
                    mean_force, std_force, rising_rate, peak_force, fft_data, max_fft_length)
                if features is not None:
                    print("特征准备成功，开始进行葡萄类型预测...")
                    prediction = predict_grape_type(features, model, weights)
                    if prediction is not None:
                        print("预测结果不为 None，设置预测结果...")
                        prediction_result = prediction
                        if prediction == 0:
                            # 当预测为 RIPE 时，执行新增的机械臂操作代码
                            threading.Thread(target=ripe_arm_operation).start()
                        elif prediction == 1:
                            # 当预测为 ROTTEN 时，执行新的机械臂操作代码
                            threading.Thread(target=rotten_arm_operation).start()
                    else:
                        print("预测结果为 None")
                else:
                    print("特征准备失败")
            else:
                print("数据提取和统计计算失败")
        else:
            print("数据不足，无法进行预测")

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, setpoint, current_value):
        error = setpoint - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# 新增：RIPE 状态下的机械臂操作
def ripe_arm_operation():
    try:
        AK = ArmIK()
        # 目标条件：最新有效力值与首次有效力值之差超过0.3 N
        threshold_delta = 0.08  
        # 从400开始，因为我们期望抓取脉冲在400~500之间
        current_pulse = 400

        # 采集空载状态下的一段数据作为基线
        baseline_samples = [read_force_sensor() for _ in range(10)]
        baseline = np.mean(baseline_samples)
        print(f"采集到的基线力值: {baseline:.3f} N")
        
        # 初始化EMA，采用较低的alpha使得平滑效果更明显
        ema_force = baseline
        alpha = 0.1  

        first_effective = None  # 保存第一次循环得到的有效力值
        
        # 循环更新EMA并计算有效力值 = ema_force - baseline
        while True:
            current_force = read_force_sensor()
            ema_force = alpha * current_force + (1 - alpha) * ema_force
            effective_force = ema_force - baseline
            if first_effective is None:
                first_effective = effective_force
            print(f"当前力值: {current_force:.3f} N, 平滑力值: {ema_force:.3f} N, 有效力值: {effective_force:.3f} N, 当前脉冲: {current_pulse}")
            
            # 当最新有效力值与首次有效力值之差超过0.3时退出
            if effective_force - first_effective >= threshold_delta:
                print("有效力值上升超过0.3 N，准备跳出循环")
                break
            
            # 增加脉冲值，每次增加10，限制不超过500
            current_pulse += 10
            if current_pulse > 500:
                current_pulse = 500
                print("脉冲值已达到500上限")
                break
            
            setBusServoPulse(1, current_pulse, 500)
            time.sleep(0.15)  # 延时以稳定数据



        RIPE_PULSE = current_pulse  # 将此时的脉冲值设为RIPE_PULSE
        print(f"最终使用的脉冲值 RIPE_PULSE: {RIPE_PULSE}")
        setBusServoPulse(1, RIPE_PULSE, 500)
        print("执行 RIPE 状态下的机械臂操作")
        time.sleep(1)
        setBusServoPulse(3, 60, 2000)
        setBusServoPulse(4, 650, 2000)
        setBusServoPulse(5, 350, 2000)
        setBusServoPulse(2, 900, 2000)
        setBusServoPulse(6, 500, 2000)
        time.sleep(2)
        setBusServoPulse(1, RIPE_PULSE, 2000)
        time.sleep(2)
        setBusServoPulse(2, 500, 2000)
        setBusServoPulse(3, 850, 2000)
        setBusServoPulse(4, 180, 2000)
        setBusServoPulse(5, 300, 2000)
        time.sleep(2)
        setBusServoPulse(1, 200, 500)
        time.sleep(1)
        setBusServoPulse(3, 60, 2000)
        setBusServoPulse(4, 650, 2000)
        setBusServoPulse(5, 350, 2000)
        setBusServoPulse(2, 900, 2000)
        setBusServoPulse(6, 500, 2000)
    except Exception as e:
        print(f"ripe_arm_operation 函数内部出错: {str(e)}")


# 新增：假设的读取力传感器函数，需根据实际硬件实现
def read_force_sensor():
    try:
        # 这里是模拟的力传感器读取代码，实际需根据硬件实现
        data = bus.read_i2c_block_data(address, 0x00, 2)
        raw = (int(data[0]) << 8) | int(data[1])  # 确保操作数为整数
        voltage = raw * 4.096 / 32767
        return voltage_to_force(voltage)
    except Exception as e:
        print(f"read_force_sensor 函数出错: {str(e)}")
        return 0

# 新增：ROTTEN 状态下的机械臂操作
def rotten_arm_operation():
    AK = ArmIK()
    setBusServoPulse(1, 200, 500)
    time.sleep(1)
    setBusServoPulse(3, 60, 1000)
    setBusServoPulse(4, 650, 1000)
    setBusServoPulse(5, 350, 1000)
    setBusServoPulse(2, 900, 1000)
    setBusServoPulse(6, 500, 1000)
    time.sleep(2)
    setBusServoPulse(1, 500, 1000)
    time.sleep(1)
    setBusServoPulse(2, 500, 1000)
    setBusServoPulse(3, 80, 1000)
    setBusServoPulse(4, 650, 1000)
    setBusServoPulse(5, 450, 1000)
    setBusServoPulse(6, 300, 1000)
    time.sleep(1)
    setBusServoPulse(6, 120, 500)
    time.sleep(1)
    setBusServoPulse(1, 200, 500)
    time.sleep(1)
    setBusServoPulse(3, 60, 1000)
    setBusServoPulse(4, 650, 1000)
    setBusServoPulse(5, 350, 1000)
    setBusServoPulse(2, 900, 1000)
    setBusServoPulse(6, 500, 1000)

# ---- 按钮点击回调 ----
def toggle_recording(event):
    global recording, recording_start_time, current_csv_file, current_csv_writer
    with csv_lock:
        if not recording:
            # 开始新记录
            recording = True
            recording_start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{timestamp}_grape.csv"

            with data_lock:
                raw_x_data.clear()
                raw_y_data.clear()
                smooth_x_data.clear()
                smooth_y_data.clear()
                maturity_text.set_text('')  # 清空成熟度文本，开始新的采集
                arm_finished_event.clear()  # 重置机械臂结束事件

            current_csv_file = open(csv_filename, 'w', newline='')
            current_csv_writer = csv.writer(current_csv_file)
            current_csv_writer.writerow(["Time", "Voltage", "Force"])

            # 启动机械臂线程
            threading.Thread(target=arm_operation, daemon=True).start()
            # 启动成熟度预测线程
            threading.Thread(target=predict_maturity, daemon=True).start()
            record_button.label.set_text('Stop')
        else:
            # 停止记录
            recording = False
            if current_csv_file:
                current_csv_file.close()
                current_csv_writer = None
                current_csv_file = None
            record_button.label.set_text('Run')
        plt.draw()

# ---- 主程序 ----
if __name__ == "__main__":
    # 绑定按钮事件
    record_button.on_clicked(toggle_recording)

    # 启动数据采集线程
    threading.Thread(target=collect_data, daemon=True).start()

    # 配置动画
    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=50,
        blit=True,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
    print("程序正常退出")

