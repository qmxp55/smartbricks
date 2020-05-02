import kivy
kivy.require('1.11.1')

from kivy.app import App
import numpy as np
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from brickcreator import SmartBricks
import shutil
from pathlib import Path
from kivy.uix.popup import Popup
from kivy.uix.button import Button
import os


class InitWindow(Screen):
    pass

class NewWindow(Screen):
    pass

class GalleryWindow(Screen):

    entry = ObjectProperty(None)

    def selected(self,filename):
        try:
            self.ids.image.source = filename[0]
            #self.ids.sel_image.text = filename[0]
        except:
            pass

outdir_ = '/home/omar/myproj_out/tmp'
outdir = Path(outdir_)
if outdir.exists() & outdir.is_dir():
    shutil.rmtree(outdir)

class SetupWindow(Screen):

    galleryid = ObjectProperty(None)
    sel_image = ObjectProperty(None)

    #def __init__(self):


    def on_enter(self, *args):
        self.sel_image.text = self.galleryid.entry.text

    def createBrick(self, imgpath, Ncolors, lowsize):

        SB = SmartBricks(imgpath=imgpath, Ncolors=np.int(Ncolors), lowsize=np.int(lowsize), outdir=outdir_)
        SB.saveProj()

class ResultWindow(Screen):

    image = ObjectProperty(None)
    #save_btn2 = ObjectProperty(None)

    #def __init__(self):
        #self.sd = SetupWindow()

    def on_enter(self, *args):
        #print(outdir_)
        self.ids.image.source = outdir_+'/'+'all'+'.jpeg'
        #self.save_btn2.bind(on_release=self.show_popup())

    def show_popup(self):
        self.show = P()
        popup = Popup(title="save current project", content=self.show, size_hint=(None,None),size=(400,400))
        popup.open()
        #show_cancel = Button(text='Cancel', size_hint_y=None, height=40)
        #show.add_widget(show_cancel)
        #show_cancel.bind(on_release=popup.dismiss)
        #print(show.pname.text)
        #print(type(show.pname.text))

        self.show.cancel_btn.bind(on_release=popup.dismiss)

        #show_save = Button(text='SAVE', size_hint_y=None, height=40)
        #self.show.add_widget(show_save)
        self.show.save_btn.bind(on_press=self.pressed)
        self.show.save_btn.bind(on_release=popup.dismiss)

    def pressed(self, instance):

        #print('IN SAVEPROJECT!!!!!!')
        #if len(savedir) > 0:
        dest = '/home/omar/myproj_out/'+self.show.pname.text
        print(dest)
        #os.makedirs(dest, exist_ok=True)
        shutil.copytree(outdir, dest)

        #dest_ = '/home/omar/myproj_out/'+savedir
        #dest = Path(dest_)
        #if dest.exists() & dest.is_dir():
        #    shutil.rmtree(dest)
        #shutil.copytree(outdir_, dest_)
        #outdir_ = dest_

class P(Screen):

    cancel_btn = ObjectProperty(None)
    save_btn = ObjectProperty(None)
    pname = ObjectProperty(None)



class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("smartbricks.kv")

class SmartBricksApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    #if using a .kv file it has to be named as the main class i.e., SmartBricksApp then smartbricks.kv.o

    SmartBricksApp().run()
