from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.factory import Factory
from kivy.properties import ListProperty, StringProperty, ObjectProperty
from kivy.core.window import Window
from kivy.uix.button import Button

from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.toast import toast
from kivymd.utils.cropimage import crop_image
from kivymd.uix.imagelist import SmartTileWithLabel
from kivymd.uix.button import MDFloatingActionButton, MDRoundFlatIconButton, MDIconButton, MDFillRoundFlatIconButton, MDFlatButton
from kivymd.uix.chip import MDChip, MDChooseChip
from kivymd.uix.dialog import MDDialog
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.bottomsheet import (
    MDCustomBottomSheet,
    MDGridBottomSheet,
    MDListBottomSheet,
)

import os
import numpy as np
import pandas as pd

from brickcreator import SmartBricks


class ContentNavigationDrawer(BoxLayout):
    pass

class CustomSmartTileWithLabel(SmartTileWithLabel):
    #stlab = ObjectProperty()
    pass

#class ContentCustomSheet(BoxLayout):
#    pass

class ItemDrawer(OneLineIconListItem):
    #icon = StringProperty()
    #target = StringProperty()
    pass


class DrawerList(ThemableBehavior, MDList):
    def set_color_item(self, instance_item):
        """Called when tap on a menu item."""

        # Set the color of the icon and text for the menu item.
        for item in self.children:
            if item.text_color == self.theme_cls.primary_color:
                item.text_color = self.theme_cls.text_color
                break
        instance_item.text_color = self.theme_cls.primary_color

class CustomMDIconButton(MDFillRoundFlatIconButton):

    #palette = ObjectProperty(None)

    color = ListProperty()
    text = StringProperty()
    icon = StringProperty()
    text_color = ListProperty()

class CustomMDChip(MDChip):

    label = StringProperty()
    cb = ObjectProperty(None)
    icon = StringProperty()


class SmartBricksApp(MDApp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            previous=True,
        )

    def build(self):
        self.screen = Builder.load_file("main.kv")

        return self.screen

    custom_sheet = None
    df = pd.read_csv('/home/omar/myproj/SmartBricks/legoKeys.cvs', sep='\t')
    out_dir = '/home/omar/myproj_out/'

    isdir = os.path.isdir(out_dir)
    if isdir:
        plist = os.listdir(out_dir)
    else:
        os.mkdir(out_dir)
        plist = None

    def callback_for_menu_items(self, *args):
        toast(args[0])

    def openScreen(self, itemdrawer):
        self.openScreenName(itemdrawer.target)
        self.root.ids.nav_drawer.set_state("close")

    def openScreenName(self, screenName):
        self.root.ids.screen_manager.current = screenName

    def ifproject(self):
        if ((self.isdir) & (len(self.plist) > 0)):
            self.openScreenName('projects')
        else:
            self.callback_for_menu_items('No projects saved yet. Select START to start a new project.')

    def chooseproject(self, img_source):

        #clear widgets if any previously load to avoid staking of color palettes between projects
        self.root.ids.content_drawer.ids.md_list.clear_widgets()

        self.root.ids.image.source = img_source
        self.openScreenName('main')

        self.img_path = os.path.dirname(img_source)
        self.tab = np.load(self.img_path+'/table.npy')

        for i in self.tab.T[0][:-1]:
            if i == '0.0_0.0_0.0': continue
            R,G,B = i.split("_")
            R,G,B = [np.int(np.float(j)) for j in [R,G,B]]
            mask = (self.df['R'] == R) & (self.df['G'] == G) & (self.df['B'] == B)
            Nbrick, color_name = self.df['LEGO No.'][mask].item(), self.df['Color'][mask].item()
            #print(R,G,B, Nbrick, color_name)

            #
            self.root.ids.content_drawer.ids.md_list.add_widget(
                CustomMDIconButton(color=(R/256,G/256,B/256,1),
                                   text=color_name,
                                   text_color=(1,0,0,1),
                                   icon='checkbox-blank-circle-outline'
                                   ))

        self.root.ids.content_drawer.ids.md_list.add_widget(
                CustomMDIconButton(color=self.theme_cls.primary_color,
                                   text='All',
                                   icon='checkbox-blank-circle-outline',
                                   #text_color=(R/256,G/256,B/256,1),
                                   #icon='checkbox-marked-circle'
                                   ))

        self.root.ids.content_drawer.ids.md_list.add_widget(
                CustomMDIconButton(color=self.theme_cls.primary_color,
                                   text='Original',
                                   icon=''
                                   ))

        keep = self.tab.T[0] == 'total'

        b2x2 = self.root.ids.brick_2x2
        b2x1 = self.root.ids.brick_2x1
        b1x1 = self.root.ids.brick_1x1
        for num, brick, brickLab in zip([1,2,3], [b2x2, b2x1, b1x1], ['2x2', '2x1', '1x1']):
            brick.text = brickLab+': '+self.tab.T[num][keep][0]


    def choose_palette(self, img_bg_color, text, id):

        self.root.ids.palette_toolbar.md_bg_color = img_bg_color

        self.img_path = os.path.dirname(self.root.ids.image.source)
        R,G,B = np.float(img_bg_color[0]*256), np.float(img_bg_color[1]*256), np.float(img_bg_color[2]*256)
        mean = np.mean([R,G,B])
        #print(R,G,B)
        if text not in ['All', 'Original']:
            self.root.ids.image.source = self.img_path+'/%s_%s_%s.gif' %(str(R), str(G), str(B))
        else:
            self.root.ids.image.source = self.img_path+'/%s.jpeg' %(text.lower())

        id.icon = 'checkbox-marked-circle'


        #Get bricks counts
        self.tab = np.load(self.img_path+'/table.npy')
        if text not in ['All', 'Original']:
            keep = self.tab.T[0] == '%s_%s_%s' %(str(R), str(G), str(B))
        else:
            keep = self.tab.T[0] == 'total'

        b2x2 = self.root.ids.brick_2x2
        b2x1 = self.root.ids.brick_2x1
        b1x1 = self.root.ids.brick_1x1

        if mean > 180:
            test_color = [0,0,0,1] #black
            invert = False
        else:
            test_color = [1,1,1,1] #withe
            invert = True

        id.text_color = test_color

        for num, brick, brickLab in zip([1,2,3], [b2x2, b2x1, b1x1], ['2x2', '2x1', '1x1']):
            brick.text = brickLab+': '+self.tab.T[num][keep][0]
            brick.text_color=test_color
            #print(brick.text_color)


        b2x2_i = self.root.ids.brick_2x2_icon
        b2x1_i = self.root.ids.brick_2x1_icon
        b1x1_i = self.root.ids.brick_1x1_icon

        for num, brick, brickLab in zip([1,2,3], [b2x2_i, b2x1_i, b1x1_i], ['2x2', '2x1', '1x1']):

            if invert: brick.icon="/home/omar/myproj/SmartBricks/%s_invert.jpg" %(brickLab)
            else: brick.icon="/home/omar/myproj/SmartBricks/%s.jpg" %(brickLab)

        #self.root.ids.brick_2x2.text = '2x2: '+self.tab.T[1][keep][0]
        #self.root.ids.brick_2x1.text = '2x1: '+self.tab.T[2][keep][0]
        #self.root.ids.brick_1x1.text = '1x1: '+self.tab.T[3][keep][0]




        #print(mean, test_color, self.root.ids.brick_2x2.text_color)
        #self.root.ids.brick_2x2.text_color=test_color

        #if invert: self.root.ids.brick_2x2_icon.icon="/home/omar/myproj/SmartBricks/2x2_invert.jpg"
        #else: self.root.ids.brick_2x2_icon.icon="/home/omar/myproj/SmartBricks/2x2.jpg"
        #print(self.root.ids.brick_2x2.text_color)




    #def test(self, R, G, B):
    #    return R,G,B,1

    def callback_mosaic_size(self, instance, value):
        toast('mosaic size: %s' %(value))
        self.mosaic_size_val = value

    def callback_mosaic_color(self, instance, value):
        toast('mosaic colors: %s' %(value))
        self.mosaic_color_val = value

    def callback_mosaic_name(self, instance, value):
        toast('mosaic name: %s' %(value))
        self.mosaic_name = value

    def show_alert_dialog(self, name=None, imgpath=None, Ncolors=None, lowsize=None, outdir=None):
            if not self.dialog:
                self.dialog = MDDialog(
                    text="Project %s already exist. Do you want to replace existing project?" %(name),
                    buttons=[
                        MDFlatButton(
                            text="CANCEL", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                        ),
                        MDFlatButton(
                            text="ACCEPT", text_color=self.theme_cls.primary_color, on_press=lambda x:self.run_mosaic(imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize, outdir=outdir)
                        ),
                    ],
                )
            self.dialog.open()

    def create_mosaic(self):

        if (self.mosaic_size_val is None) or (self.mosaic_color_val is None):
            toast('Choose mosaic size and colors first')
        else:
            #print(self.root.ids.setup_image.source)
            #print(int(self.mosaic_size_val))
            #print(int(self.mosaic_color_val))
            #print(self.root.ids.project_name.text)
            #print(self.out_dir+self.root.ids.project_name.text)

            imgpath = str(self.root.ids.setup_image.source)
            Ncolors = np.int(self.mosaic_color_val)
            lowsize = np.int(self.mosaic_size_val)
            outdir = str(self.out_dir + self.root.ids.project_name.text)

            for i in [imgpath, Ncolors, lowsize, outdir]:
                print(i, type(i))

        if (self.plist is not None) & (self.root.ids.project_name.text in self.plist):

            print('project name already exist...')
            self.show_alert_dialog(name=self.root.ids.project_name.text, imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize, outdir=outdir)

        else:
            self.run_mosaic(imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize, outdir=outdir)
            #print('HERE!!!!!!!')
            #SB = SmartBricks(imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize, outdir=outdir)
            #SB.saveProj()



    def run_mosaic(self, imgpath=None, Ncolors=None, lowsize=None, outdir=None):

        #print(imgpath, Ncolors, lowsize, outdir)
        #if self.dialog:
        #    self.dialog.dismiss()

        SB = SmartBricks(imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize,
                             outdir=outdir)
        SB.saveProj()

        self.chooseproject(outdir+'/all.jpeg')




    def on_start(self):

        self.mosaic_size_val = None
        self.mosaic_color_val = None
        self.mosaic_name = None
        self.dialog = None

        for i in np.arange(8,72,8):
            self.root.ids.mosaic_size.add_widget(
                    CustomMDChip(label=str(i), cb=self.callback_mosaic_size, icon='grid'))

        for i in np.arange(2,18,2):
            self.root.ids.mosaic_colors.add_widget(
                    CustomMDChip(label=str(i), cb=self.callback_mosaic_color, icon='palette'))

        if ((self.isdir) & (len(self.plist) > 0)):
            for dir in self.plist:
                self.root.ids.grid_list.add_widget(
                    CustomSmartTileWithLabel(source = self.out_dir+dir+'/all.jpeg',
                                             text = "[size=32]%s[/size]" %(dir))
                )

            #print(self.custbutt.palette.md_bg_color)

            #self.root.ids.content_drawer.ids.palette.md_bg_color = (i/100,i/10,0,1)
            #md_bg_color=(1,0,0,1)

            #

#        self.root.ids.content_drawer.ids.md_list.add_widget(
#            ItemDrawer(target="screen1", text="Screen 1",
#                       icon="home-circle-outline",
#                       on_release=self.openScreen)
#        )
#        self.root.ids.content_drawer.ids.md_list.add_widget(
#            ItemDrawer(target="screen2", text="Screen 2",
#                       icon="settings-outline",
#                       on_release=self.openScreen)
#        )


    def custom_bottom_sheet(self):
        self.custom_sheet = MDCustomBottomSheet(screen=Factory.ContentCustomSheet())
        self.custom_sheet.open()

    def file_manager_open(self):
        self.file_manager.show('/home/omar/Pictures')  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        self.openScreenName('setup')
        self.root.ids.setup_image.source = path
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True


'''

    def crop_image_for_tile(self, instance, size, path_to_crop_image):
        if not os.path.exists(os.path.join(self.directory, path_to_crop_image)):
            size = (int(size[0]), int(size[1]))
            path_to_origin_image = path_to_crop_image.replace("_tile_crop", "")
            crop_image(size, path_to_origin_image, path_to_crop_image)
        instance.source = path_to_crop_image


    def show_example_bottom_sheet(self):
        bs_menu = MDListBottomSheet()
        bs_menu.add_item(
            "Select Image from", lambda x: self.callback_for_menu_items("Here's an item with an icon"))
        bs_menu.add_item(
            "CAMERA",
            lambda x: self.callback_for_menu_items("Here's an item with an icon"),
            icon="clipboard-account",
        )
        bs_menu.add_item(
            "GALLERY",
            lambda x: self.openScreenName('screen3'),
            icon="nfc",
        )
        bs_menu.open()
'''


if __name__ == "__main__":
    SmartBricksApp().run()
