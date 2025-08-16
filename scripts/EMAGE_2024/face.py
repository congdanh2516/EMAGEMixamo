import socket
import time
import threading
import struct
import json
import wave

def print_current_millis():
    millis = int(time.time() * 1000)
    print(f"Current time: {millis} ms")
    

HOST = '127.0.0.1'  
PORT = 5005  
AUDIO_PATH = "C:\\Users\\Duy\\Downloads\\UnityControl\\demo_face\\1_fufu_0_7_7.wav"  # đoạn 15s WAV gốc
CHUNK_DURATION = (1/30.0)*64  # số giây muốn cắt ra


def split_audio_chunks(path, chunk_duration_sec):
    with wave.open(path, 'rb') as wav:
        frame_rate = wav.getframerate()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        total_frames = wav.getnframes()

        frames_per_chunk = int(frame_rate * chunk_duration_sec)
        num_chunks = total_frames // frames_per_chunk

        audio_chunks = []
        for i in range(num_chunks):
            wav.setpos(i * frames_per_chunk)
            chunk = wav.readframes(frames_per_chunk)
            audio_chunks.append(chunk)

    return audio_chunks, frame_rate, num_channels, sample_width

def client_handler(conn, addr):
    """Handles individual client connections."""
    print(f"Connected by {addr}")
    try:
        time.sleep(2)
        with open("C:\\Users\\Duy\\Downloads\\UnityControl\\clip_17_merged.json", "r") as file:
            full_list = json.load(file)
        print(type(full_list))
        frame0 = full_list[0] # First frame contains T-pose data

        # with open("C:/Users/Duy/Downloads/UnityControl/offset_output_demo.json", "r") as file:
        #     offsets = json.load(file)

        # frame0["offsets"] = offsets  # Add offsets to the first frame

        json_str = json.dumps([frame0])
        json_bytes = json_str.encode('utf-8')
        length_prefix = struct.pack('<I', len(json_bytes))
        conn.sendall(length_prefix + json_bytes)
        print("Sent frame 0 (T-pose) to", addr)

        # chia audio ra thành nhiều chunk
        audio_chunks, sample_rate, num_channels, sample_width = split_audio_chunks(AUDIO_PATH, CHUNK_DURATION)
        total_audio_chunks = len(audio_chunks)
        audio_sent = 0
        
        # dữ liệu từ file 
        with open("C:\\Users\\Duy\\Downloads\\UnityControl\\demo_face\\clipped_result_structured.json", "r") as file:
            demo_face_data = json.load(file)
            temp_l = []
            blendshape_names = demo_face_data["name"]
            # blendshape_names = blendshape_names[1:]
            frames = demo_face_data["frames"]
            az = 0
            for frame in frames:
                weights = frame["weights"]
                merged_weights = dict(zip(blendshape_names, weights))
                blendshape = {"Newton_HeadFace": merged_weights}
                blendshape = {"blendshape": blendshape}
                temp_l.append(blendshape)
                if az ==0:
                    print(f"Blendshape names: {blendshape}")
                    az += 1
            full_list = temp_l
        time.sleep(2)
        # full_list = full_list[7:] # Bỏ qua 7 frame đầu tiên (T-pose + 6 frame đầu tiên) because audio start 0.2s later.
        batch_size = 64
        total = len(full_list)
        print(f"Total frames to send: {total}, Total audio chunks: {total_audio_chunks}")
        start_idx = 0
        print(f"Total items to send: {total}")
        while start_idx < total:
            end_idx = min(start_idx + batch_size, total)
            batch = full_list[start_idx:end_idx]
            json_str = json.dumps(batch)
            json_bytes = json_str.encode('utf-8')
            length_prefix = struct.pack('<I', len(json_bytes))  # little-endian 4-byte length prefix
            conn.sendall(length_prefix + json_bytes)
            print(f"Sent batch {start_idx} to {end_idx - 1} ({len(json_bytes)} bytes) to {addr}")
            start_idx += batch_size

            # GỬI THÊM AUDIO 2s nếu còn
            if audio_sent < total_audio_chunks:
                chunk = audio_chunks[audio_sent]
                conn.sendall(b'AUDP')
                conn.sendall(struct.pack('<III', sample_rate, num_channels, sample_width))
                conn.sendall(struct.pack('<I', len(chunk)))
                conn.sendall(chunk)
                print(f"Sent audio chunk #{audio_sent+1}/{total_audio_chunks}")
                audio_sent += 1
            time.sleep((1/30.0)*64)  # wait 2 seconds between batches
    except ConnectionResetError:
        print(f"Client {addr} disconnected unexpectedly.")
    except BrokenPipeError:
        print(f"Client {addr} disconnected (broken pipe).")
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        print(f"Closing connection with {addr}")
        conn.close()

def start_server():
    """Starts the TCP socket server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen() # Listen for incoming connections
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept() # Accept a new connection
            # Start a new thread to handle the client
            client_thread = threading.Thread(target=client_handler, args=(conn, addr))
            client_thread.start()
            print(f"Active connections: {threading.active_count() - 1}")

if __name__ == "__main__":
    start_server()