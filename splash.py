from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk

class AnimatedGIF(Label, object):
    def __init__(self, master, path, forever=True):
        self._master = master
        self._loc = 0
        self._forever = forever

        self._is_running = False

        im = Image.open(path)
        self._frames = []
        i = 0
        try:
            while True:
                photoframe = ImageTk.PhotoImage(im.copy().convert('RGBA'))
                self._frames.append(photoframe)

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._last_index = len(self._frames) - 1

        self._image_width, self._image_height = im.size

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 100

        self._callback_id = None

        super(AnimatedGIF, self).__init__(master, image=self._frames[0])

    def getWidth(self):
        return self._image_width

    def getHeight(self):
        return self._image_height

    def start_animation(self, frame=None):
        if self._is_running: return

        if frame is not None:
            self._loc = 0
            self.configure(image=self._frames[frame])

        self._master.after(self._delay, self._animate_GIF)
        self._is_running = True

    def stop_animation(self):
        if not self._is_running: return

        if self._callback_id is not None:
            self.after_cancel(self._callback_id)
            self._callback_id = None

        self._is_running = False

    def _animate_GIF(self):
        if (len(self._frames) == 1):
            working = False
            
        else:
            self._loc += 1
            self.configure(image=self._frames[self._loc])

            if self._loc == self._last_index:
                if self._forever:
                    self._loc = 0
                    self._callback_id = self._master.after(self._delay, self._animate_GIF)
                    self.master.destroy()
                else:
                    self._callback_id = None
                    self._is_running = False
            else:
                self._callback_id = self._master.after(self._delay, self._animate_GIF)

    def pack(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).pack(**kwargs)

    def grid(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).grid(**kwargs)

    def place(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).place(**kwargs)

    def pack_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).pack_forget(**kwargs)

    def grid_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).grid_forget(**kwargs)

    def place_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).place_forget(**kwargs)

root = Tk()
l = AnimatedGIF(root, "Green.PNG")
l.pack()

ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
w = l.getWidth()
h = l.getHeight()
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)

root.geometry('%dx%d+%d+%d' % (w, h+30, x, y))
root.overrideredirect(True)

progressbar = ttk.Progressbar(root, orient=HORIZONTAL, length=8000, mode='determinate')
progressbar.pack(side="bottom")
progressbar.start()

root.after(12000, root.destroy)
root.attributes('-topmost', 'true')
root.overrideredirect(True)

root.mainloop()


