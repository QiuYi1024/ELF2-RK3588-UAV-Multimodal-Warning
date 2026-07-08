import pyaudio
import wave
import sys
import numpy as np

def find_hardware_device(p):
    """遍历所有音频设备，寻找直通硬件的通道 (绕过 Ubuntu 系统混音器)"""
    target_index = None
    fallback_index = None
    
    print("\n🔍 正在扫描系统底层音频接口...")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev.get('name', '')
        
        # 如果找到 6 通道 USB/阵列类输入设备，先记下来做备用。
        lower_name = name.lower()
        if dev.get("maxInputChannels", 0) >= 6 and any(token in lower_name for token in ("usb", "array", "mic", "microphone")):
            fallback_index = i
            
        # 最完美的通道：带有 hw: 开头的底层硬件节点
        if fallback_index == i and ("hw:" in name or "plughw:" in name):
            target_index = i
            print(f"✅ 锁定最高级硬件直通通道: [{i}] {name}")
            break
            
    if target_index is None and fallback_index is not None:
        target_index = fallback_index
        print(f"⚠️ 未找到 hw: 直通节点，使用默认节点: [{target_index}]")
        
    return target_index

def record_hw_bypass(record_seconds=8):
    RATE = 16000
    CHANNELS = 6
    CHUNK = 1024

    p = pyaudio.PyAudio()
    dev_index = find_hardware_device(p)

    if dev_index is None:
        print("❌ 找不到 6 通道 USB 麦克风阵列，请检查 USB 连接！")
        sys.exit(1)

    print("\n🎙️ 准备开始底通信录音...")
    
    try:
        # 指定 input_device_index，强制从底层节点提货！
        stream = p.open(
            rate=RATE,
            format=pyaudio.paInt16,
            channels=CHANNELS,
            input=True,
            input_device_index=dev_index,
            frames_per_buffer=CHUNK
        )
    except OSError as e:
        print(f"\n❌ 声卡独占打开失败: {e}")
        sys.exit(1)

    print("🔴 正在录音，请对麦克风说话！")
    
    ch0_frames = []
    ch1_frames = []

    for _ in range(int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        matrix = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        
        # 提取 CH0 (DSP原生)
        ch0_frames.append(matrix[:, 0].tobytes())
        # 提取 CH1 (原声，放大 20 倍)
        ch1_amp = (matrix[:, 1].astype(np.float32) * 20.0)
        np.clip(ch1_amp, -32768, 32767, out=ch1_amp)
        ch1_frames.append(ch1_amp.astype(np.int16).tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存两个文件对比
    for filename, frames in [("hw_ch0_dsp.wav", ch0_frames), ("hw_ch1_raw.wav", ch1_frames)]:
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    print("\n✅ 录制完成！生成了 hw_ch0_dsp.wav 和 hw_ch1_raw.wav")
    print("💡 赶快去听一听！只要这两个里有一个能出人声，我们的大业就成了！")

if __name__ == "__main__":
    record_hw_bypass()
