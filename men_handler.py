import ctypes

import win32api
import win32con
import win32process
from mem_edit import Process

from ctypes import *
from ctypes.wintypes import *

import psutil
class MemHandler:

    def __init__(self):
        # Open the process
        self.p_id, self.base_address = self.get_process_by_name("th16")
        # The static address needs the program base_address
        self.address = self.base_address + 0x00571160


    def get_mem(self, offsets):
        pointer_value, value = self.read_process_memory(self.p_id, self.address, offsets)
        return value
        #print(f"(Static Address) Value: {value}")

        ## Re-reading the memory with the last pointer
        # pointer_value, value = read_process_memory(p_id, pointer_value, None)
        # print(f"(Dynamic Address) Value: {value}")

    def get_process_by_name(self, process_name):
        """Finds the process id of the given
        process name and returns the process id and its base address."""

        process_name = process_name.lower()

        # Enumerate all processes
        processes = win32process.EnumProcesses()

        for process_id in processes:
            # If process_id is the same as this program, skip it
            if process_id == -1:
                continue

            # Try to read the process memory
            try:
                h_process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, True, process_id)

                # Try to read the modules of the process
                try:
                    # modules is an array of base addresses of each module
                    modules = win32process.EnumProcessModules(h_process)

                    for base_address in modules:
                        # Get the name of the module
                        name = str(win32process.GetModuleFileNameEx(h_process, base_address))

                        # Compare it to the name of your program
                        if name.lower().find(process_name) != -1:
                            return process_id, base_address
                finally:
                    win32api.CloseHandle(h_process)
            except:
                pass

    def read_process_memory(self, process_id, address, offsets=[]):
        """Read a process' memory based on its process id, address and offsets.
        Returns the address without offsets and the value."""

        # The handle to the program's process
        # This will allow to use ReadProcessMemory
        h_process = ctypes.windll.kernel32.OpenProcess(win32con.PROCESS_VM_READ, False, self.p_id)

        # This is a pointer to the data you want to read
        # Use `data.value` to get the value at this pointer
        # In this case, this value is an Integer with 4 bytes
        data = ctypes.c_uint(0)

        # Size of the variable, it usually is 4 bytes
        bytesRead = ctypes.c_uint(0)

        # Starting address
        current_address = address

        if offsets:
            # Append a new element to the offsets array
            # This will allow you to get the value at the last offset
            offsets.append(None)

            for offset in offsets:
                # Read the memory of current address using ReadProcessMemory
                ctypes.windll.kernel32.ReadProcessMemory(h_process, current_address, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytesRead))

                # If current offset is `None`, return the value of the last offset
                if not offset:
                    return current_address, data.value
                else:
                    # Replace the address with the new data address
                    current_address = data.value + offset

        else:
            # Just read the single memory address
            ctypes.windll.kernel32.ReadProcessMemory(h_process, current_address, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytesRead))

        # Close the handle to the process
        ctypes.windll.kernel32.CloseHandle(h_process)

        # Return a pointer to the value and the value
        # The pointer will be used to write to the memory
        return current_address, data.value

'''tmp = MemHandler()
print(tmp.get_mem(offsets=[0x4A57f4]))
print(tmp.get_mem(offsets=[0x4A5800]))
print(tmp.get_mem(offsets=[0x4A57C0]))'''
