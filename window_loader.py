import numpy as np
import win32con
import win32gui
import win32ui
from pywinauto import Desktop


class WindowLoader:

    def __init__(self, title='Hidden Star in Four Seasons'):
        windows = Desktop(backend="uia").windows()
        for x in [w.window_text() for w in windows]:
            if title in x:
                self.title = x
        hwnd = win32gui.FindWindow(None, self.title)
        win32gui.SetForegroundWindow(hwnd)

    def get_window(self):
        hwnd = win32gui.FindWindow(None, self.title)
        win32gui.SetForegroundWindow(hwnd)
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        width = 640
        height = 480
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        while (True):
            cDC.BitBlt((0, 0), (width, height), dcObj, (0 + 4, 0 + 38), win32con.SRCCOPY)
            # dataBitMap.SaveBitmapFile(cDC, 'screenshot.bmp')
            yield np.frombuffer(dataBitMap.GetBitmapBits(True), dtype='uint8')

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

    def close(self):
        #self.dcObj.DeleteDC()
        #self.cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wDC)
        win32gui.DeleteObject(self.dataBitMap.GetHandle())
