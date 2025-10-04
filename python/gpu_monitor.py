# Copyright 2025 Joseph Hersey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPU Monitor for Intel Arc GPUs
Shows available devices and their status
"""

from openvino.runtime import Core
import time
import os

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor_gpus(refresh_interval=2):
    """Monitor GPU devices"""
    ie = Core()
    
    print("🚀 Intel GPU Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            clear_screen()
            print("🚀 Intel GPU Monitor - Lambda Project")
            print("=" * 60)
            print(f"⏰ Refresh: {refresh_interval}s\n")
            
            devices = ie.available_devices
            
            print(f"📊 Available Devices: {len(devices)}\n")
            
            for device in devices:
                try:
                    device_name = ie.get_property(device, 'FULL_DEVICE_NAME')
                    
                    # Try to get additional properties
                    print(f"🔹 {device}")
                    print(f"   Name: {device_name}")
                    
                    # Check device type
                    if device.startswith('GPU'):
                        print(f"   Type: Intel Arc GPU")
                    elif device == 'NPU':
                        print(f"   Type: AI Boost NPU")
                    elif device == 'CPU':
                        print(f"   Type: CPU")
                    
                    print()
                    
                except Exception as e:
                    print(f"🔹 {device}: Unable to get details ({e})\n")
            
            print("=" * 60)
            print(f"💡 To use specific device in code:")
            print(f'   indexer = LambdaProjectIndexer(device="GPU.0")')
            print(f'   indexer = LambdaProjectIndexer(device="GPU.1")')
            print(f'   indexer = LambdaProjectIndexer(device="NPU")')
            print("=" * 60)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped monitoring")

def list_devices_once():
    """List all devices once and exit"""
    ie = Core()
    
    print("🚀 Intel GPU Device List")
    print("=" * 60)
    
    devices = ie.available_devices
    print(f"Found {len(devices)} device(s):\n")
    
    for device in devices:
        try:
            device_name = ie.get_property(device, 'FULL_DEVICE_NAME')
            print(f"✓ {device}: {device_name}")
        except:
            print(f"✓ {device}: (details unavailable)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        list_devices_once()
    else:
        monitor_gpus()

# EOF
