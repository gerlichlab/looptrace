from pychrtrace.image_comparison import Compare
from tkinter.filedialog import askopenfilename

def main():
    
    config_path = askopenfilename()
    C = Compare(config_path)
    print('Initialized image comparison class with config: ', C.config)
    res = C.compare_multi_sc()

if __name__ == '__main__':
    main()