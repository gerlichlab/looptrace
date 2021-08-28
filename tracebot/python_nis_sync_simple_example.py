import win32event
import time

if __name__ == '__main__':
    event_image_available = win32event.CreateEvent(None,0,0,"NIS_Image_Available")
    event_image_processed = win32event.CreateEvent(None,0,0,"NIS_Image_Processed")

    while True:
        print("Waiting for next image...")
        win32event.WaitForSingleObject(event_image_available,-1)
        time.sleep(4)   # Replace this with control functions/other communcations.
        print("Procedure done!")
        win32event.SetEvent(event_image_processed)
