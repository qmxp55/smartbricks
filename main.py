#
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.factory import Factory
from kivy.properties import ListProperty, StringProperty, ObjectProperty, NumericProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.metrics import dp


from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.toast import toast
from kivymd.uix.imagelist import SmartTileWithLabel
from kivymd.uix.button import MDFloatingActionButton, MDRoundFlatIconButton, MDIconButton, MDFillRoundFlatIconButton, MDFlatButton
from kivymd.uix.chip import MDChip, MDChooseChip
from kivymd.uix.dialog import MDDialog
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.bottomsheet import MDCustomBottomSheet, MDGridBottomSheet, MDListBottomSheet

import os
import io
import PIL
import time
import glob
import numpy as np
import pandas as pd

from brickcreator import SmartBricks

try:
    from android.storage import app_storage_path
    from android.storage import primary_external_storage_path
    from android.permissions import request_permissions, Permission
except ModuleNotFoundError:
    pass



class ContentNavigationDrawer(BoxLayout):
    pass

class CustomSmartTileWithLabel(SmartTileWithLabel):
    pass

class CustomSmartTileWithLabelGallery(SmartTileWithLabel):
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

class progress_bar(ProgressBar):

    #current_value = NumericProperty(10)
    load_bar = ObjectProperty(None)


class SmartBricksApp(MDApp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            previous=True, ext=['jpeg', 'png', 'jpg'])

        try:

            request_permissions([Permission.WRITE_EXTERNAL_STORAGE,
                             Permission.READ_EXTERNAL_STORAGE])
        except NameError:
            pass

    def build(self):
        #Window.bind(on_keyboard=self.key_input)
        self.screen = Builder.load_file("main.kv")

        return self.screen

    def on_pause(self):
        return True

    #def key_input(self, window, key, scancode, codepoint, modifier):
    #  if key == 27:
    #     return True  # override the default behaviour
    #  else:           # the key now does nothing
#         return False

    #settings_path = app_storage_path()

    path = os.getcwd()
    df = pd.read_csv('%s/legoKeys.cvs' %(path), sep='\t')

    try:
        SD_CARD = primary_external_storage_path()+'/'+'DCIM/Camera'
        out_dir = '%s/smartbricks_outputs/' %(primary_external_storage_path())
        print('ANDROID mmodules loaded...')

    except NameError:
        print('ANDROID modules failed...')
        SD_CARD = '/home/omar/Pictures'
        out_dir = '%s/myproj_out/' %(path)
        pass
    
    isdir = os.path.isdir(out_dir)

    def plist(self):
        
        if self.isdir:
            plist_ = os.listdir(self.out_dir)
        else:
            os.mkdir(self.out_dir)
            plist_ = None
            
        return plist_

    custom_sheet = None

    def callback_for_menu_items(self, *args):
        toast(args[0])

    def openScreen(self, itemdrawer):
        self.openScreenName(itemdrawer.target)
        self.root.ids.nav_drawer.set_state("close")

    def openScreenName(self, screenName):
        self.root.ids.screen_manager.current = screenName

    def ifproject(self):
        
        print('PLIST!!!!!!!!!', self.plist())
        print('PLIST TYPE!!!!!!!!!', type(self.plist()))
        
        if ((self.isdir) & (self.plist() is not None)):
            
            #clear widgets if any previously load to avoid staking of color palettes between projects
            self.root.ids.grid_list.clear_widgets()
            #self.root.ids.grid_list.remove_widget(self.root.ids.grid_list)
            
            self.openScreenName('projects')
            
            #if ((self.isdir) & (self.plist is not None)):
            for dir_ in self.plist():
                self.root.ids.grid_list.add_widget(CustomSmartTileWithLabel(source = self.out_dir+dir_+'/all.jpeg', text = "[size=32]%s[/size]" %(dir_)))
            
        else:
            self.callback_for_menu_items('No projects saved yet. Select START to start a new project.')

    def chooseproject(self, img_source):

        #clear widgets if any previously load to avoid staking of color palettes between projects
        self.root.ids.content_drawer.ids.md_list.clear_widgets()

        self.root.ids.image.source = img_source
        self.openScreenName('main')

        self.img_path = os.path.dirname(img_source)
        tab = np.load(self.img_path+'/table.npz')
        self.tab = tab['data']

        row_data = []
        n2x2, n2x1, n1x1 = 0, 0, 0

        for num, i in enumerate(self.tab.T[0][:-1]):
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

            n2x2 += int(self.tab.T[1][num])
            n2x1 += int(self.tab.T[2][num])
            n1x1 += int(self.tab.T[3][num])

            row_data.append((color_name, self.tab.T[1][num], self.tab.T[2][num], self.tab.T[3][num], int(self.tab.T[1][num])+int(self.tab.T[2][num])+int(self.tab.T[3][num])))

        row_data.append(('Total', n2x2, n2x1, n1x1, n2x2+n2x1+n1x1))
        #if len(row_data) > 10: pagination = True
        #else: pagination = False

        self.data_tables = MDDataTable(
            size_hint=(0.9, 0.6),
            rows_num=20,
            use_pagination=True if len(row_data) > 20 else False,
            check=False,
            column_data=[
                ("Color", dp(40)),
                ("2x2", dp(10)),
                ("2x1", dp(10)),
                ("1x1", dp(10)),
                ("All", dp(10))
            ],
            row_data=row_data,
        )

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
        #R,G,B = np.float(img_bg_color[0]*256), np.float(img_bg_color[1]*256), np.float(img_bg_color[2]*256)
        R,G,B = np.int(img_bg_color[0]*256), np.int(img_bg_color[1]*256), np.int(img_bg_color[2]*256)

        mean = np.mean([R,G,B])
        #print(R,G,B)
        
        
        #self.root.ids.image.parent.remove_widget(self.root.ids.image)
        
        if text not in ['All', 'Original']:
            self.root.ids.image.clear_widgets()
            self.root.ids.image.source = self.img_path+'/%s_%s_%s.zip' %(str(R), str(G), str(B))
        else:
            self.root.ids.image.source = self.img_path+'/%s.jpeg' %(text.lower())
            
        print('SOURCE!!!!!!!', self.root.ids.image.source)
        print('TEXT!!!!!!!', text)

        id.icon = 'checkbox-marked-circle'


        #Get bricks counts
        tab = np.load(self.img_path+'/table.npz')
        self.tab = tab['data']

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

            #print('MAIN PATH:', self.path)

            if invert: brick.icon="%s/images/%s_invert.jpg" %(self.path, brickLab)
            else: brick.icon="%s/images/%s.jpg" %(self.path, brickLab)

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

    def show_alert_dialog(self, name):

        if not self.dialog:
            self.dialog = MDDialog(
                title="Replace existing project?",
                text="Project '%s' already exists. Do you want to replace existing project?" %(name),
                buttons=[
                    MDFlatButton(
                        text="CANCEL", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                    ),
                    MDFlatButton(
                        text="ACCEPT", text_color=self.theme_cls.primary_color, on_press=lambda x:self.show_dialog_progress()
                    ),
                ],
            )
        else: self.dialog.dismiss()
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

            self.imgpath = str(self.root.ids.setup_image.source)
            self.Ncolors = np.int(self.mosaic_color_val)
            self.lowsize = np.int(self.mosaic_size_val)
            self.outdir = str(self.out_dir + self.root.ids.project_name.text)

            for i in [self.imgpath, self.Ncolors, self.lowsize, self.outdir]:
                print(i, type(i))



        if (self.plist() is not None):
            if (self.root.ids.project_name.text in self.plist()):

                print('project name already exist...')
                self.show_alert_dialog(name=self.root.ids.project_name.text)
            else:
                self.show_dialog_progress()
        else:
            self.show_dialog_progress()


    def show_dialog_progress(self):

        #if not self.dialog2:
        self.dialog2 = MDDialog(
                        title="Creating mosaic. Please wait.",
                        type="custom",
                        #text="Creating mosaic. Please wait.",
                        content_cls=progress_bar(),#self.pb,
                        on_open=self.run_mosaic
                        #on_open=self.puopen)
                        #on_open=self.run_mosaic(imgpath=imgpath, Ncolors=Ncolors, lowsize=lowsize, outdir=outdir)
                        )
        self.dialog2.open()

        if self.dialog: self.dialog.dismiss()
        #self.dialog2.bind(on_open=self.run_mosaic)
        #self.run_mosaic



    #def run_mosaic(self, imgpath=None, Ncolors=None, lowsize=None, outdir=None):
    def run_mosaic(self, instance):
        
        #clear widgets if any previously load to avoid staking of color palettes between projects
        #self.pb.load_bar.clear_widgets()
        self.pb = progress_bar()
        #self.pb.load_bar.parent.remove_widget(self.pb.load_bar)
        self.pb.load_bar.value = 0

        #print(self.pb.load_bar.value)
        #Nmax = np.int(self.mosaic_color_val) #+ 3

        start = time.time()
        SB = SmartBricks(imgpath=self.imgpath, Ncolors=self.Ncolors, lowsize=self.lowsize, outdir=self.outdir)
        #print(SB.img)
        #print(SB.res1x1)
        #SB.saveProj()
        end = time.time()
        print('Total run time #1: %f sec' %(end - start))
        print('point size', SB.size)


        import matplotlib.pyplot as plt
        ispathdir = os.path.isdir(self.outdir)
        if not ispathdir: os.makedirs(self.outdir, exist_ok=True)
        else:
            files = os.listdir(self.outdir)
            #print(self.outdir)
            for f in files:
                os.remove(self.outdir+'/'+f)

        start = time.time()
        
        #lmax = 10
            
        if SB.w > SB.h: x_size, y_size = SB.lmax, SB.h*SB.lmax/SB.w
        else: x_size, y_size = SB.w*SB.lmax/SB.h, SB.lmax

        fig = plt.figure(figsize=(x_size,y_size))
        #fig = plt.figure(figsize=(12,12))
        ax = plt.gca()

        SB.bricksCanvas(img=SB.img, fig=fig, ax=ax, RGB=None, res2x2=SB.res2x2, res2x1=SB.res2x1, res1x1=SB.res1x1)
        figcvs = fig
        figall = fig
        #figoriginal = fig.copy

        #paletteLego = SB.palette(SB.img)
        #palette_flat = SB.imgFlat(paletteLego)
        Nmax = len(SB.palette_flat)
        self.pb.load_bar.max = Nmax

        table = []
        #for num, pal in enumerate(palette_flat):
        for i in range(Nmax):

            print(self.pb.load_bar.value)

            pal = SB.palette_flat[i]
            N2x2, N2x1, N1x1 = SB.makeGiff(img=SB.img, RGB=pal, idxs=[SB.res2x2[2], SB.res2x1[2], SB.res1x1[2]], pathdir=self.outdir, fig=figcvs, ax=ax)
            r,g,b = pal
            color = '%s_%s_%s' %(r,g,b)
            table.append([color, N2x2, N2x1, N1x1])
            self.pb.load_bar.value = i+1
            #self.value99 = i+1

        t = np.array(table)
        N2x2total = np.sum(t[:,1].astype(int))
        N2x1total = np.sum(t[:,2].astype(int))
        N1x1total = np.sum(t[:,3].astype(int))
        table.append(['total', N2x2total, N2x1total, N1x1total])
        
        end = time.time()
        print('Total run time #2: %f sec' %(end - start))
        
        
        start = time.time()
        
        figall.subplots_adjust(left=SB.left, bottom=SB.bottom, right=SB.right, top=SB.top, wspace=None, hspace=None)
        ax = figall.add_subplot(111)
        ax.imshow(SB.img)
        #True if testing App from PC.
        if SB.frompc:
            figall.savefig('%s/all.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)
        else:
            buffer = io.BytesIO()
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            pil_image.save('%s/all.jpeg' %(self.outdir), 'JPEG')

        fig0 = plt.figure(figsize=(x_size,y_size))
        fig0.subplots_adjust(left=SB.left, bottom=SB.bottom, right=SB.right, top=SB.top, wspace=None, hspace=None)
        ax = plt.gca()
        ax = fig0.add_subplot(111)
        ax.imshow(SB.img_original)
        if SB.frompc:
            fig0.savefig('%s/original.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)
        else:
            buffer = io.BytesIO()
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            pil_image.save('%s/original.jpeg' %(self.outdir), 'JPEG')

        #ax = figall.add_subplot(111)
        #ax.imshow(SB.img)
        #figall.savefig('%s/all.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)

        #fig0 = plt.figure(figsize=(12,12))
        #ax = fig0.add_subplot(111)
        #plt.imshow(SB.img_original)
        #fig0.savefig('%s/original.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)

        np.savez_compressed('%s/table' %(self.outdir), data=table)
        
        end = time.time()
        print('Total run time #3: %f sec' %(end - start))

        if Nmax == self.pb.load_bar.value:
            self.dialog2.dismiss()

        #SmartBricks(imgpath=self.imgpath, Ncolors=self.Ncolors, lowsize=self.lowsize, outdir=self.outdir).saveProj(self.pb.load_bar.value)

        self.chooseproject(self.outdir+'/all.jpeg')

    def next(self, dt):

        #if self.times == 0:
            #self.run_mosaic
        #    SmartBricks(imgpath=self.imgpath, Ncolors=self.Ncolors, lowsize=self.lowsize, outdir=self.outdir).saveProj()
        #    self.times = 1


        if os.path.exists(self.outdir):
            #self.value99 += 1
            print(self.pb.load_bar.value)
            Nmax = np.int(self.mosaic_color_val) + 3
            self.pb.load_bar.max = Nmax
            Ncurrent = len(os.listdir(str(self.out_dir + self.root.ids.project_name.text)))
            print(self.pb.load_bar.value, self.pb.load_bar.max, Ncurrent)
            #print(self.pb.load_bar.value, self.pb.load_bar.max)


            if self.pb.load_bar.value>=Nmax:
                return False
            else:
                self.pb.load_bar.value = Ncurrent
                #self.pb.load_bar.value = self.value99
        else:
            print('PATH DOES NOT EXIST YET!')
            #print(self.times)


    def puopen(self, instance):

        #self.times = 0

        Clock.schedule_interval(self.next, 1/25)
        #self.run_mosaic

    def on_start(self):

        self.imgpath = None
        self.Ncolors = None
        self.lowsize = None
        self.outdir = None

        self.mosaic_size_val = None
        self.mosaic_color_val = None
        self.mosaic_name = None
        self.dialog = None
        self.dialog2 = None
        self.value99 = 0
        #self.pb = progress_bar()

        self.root.ids.avatar.source = '%s/images/logo.zip' %(self.path)

        self.root.ids.brick_2x2_icon.icon = "%s/images/%s.jpg" %(self.path, '2x2')
        self.root.ids.brick_2x1_icon.icon = "%s/images/%s.jpg" %(self.path, '2x1')
        self.root.ids.brick_1x1_icon.icon = "%s/images/%s.jpg" %(self.path, '1x1')

        for i in np.arange(8,72,8):
            self.root.ids.mosaic_size.add_widget(
                    CustomMDChip(label='%s' %(str(i)), cb=self.callback_mosaic_size, icon='grid'))

        for i in np.arange(2,18,2):
            self.root.ids.mosaic_colors.add_widget(
                    CustomMDChip(label=str(i), cb=self.callback_mosaic_color, icon='palette'))


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
        
    def show_gallery(self):
        
        self.openScreenName('camera')
        
        types = ('*.png', '*.jpeg', '*.jpg') # the tuple of file types
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(self.SD_CARD, files)))

        files_grabbed.sort(key=os.path.getmtime)
        files_grabbed.reverse()
        
        for file in files_grabbed[:20]:
            self.root.ids.grid_list_camera.add_widget(
                CustomSmartTileWithLabelGallery(source = file, text = "[size=18]%s[/size]" %(os.path.basename(file)))
                )

    def file_manager_open(self):
        #self.file_manager.show('/home/omar/Pictures')  # output manager to the screen
        #self.file_manager.show(self.settings_path)
        self.file_manager.show(self.SD_CARD)
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        try:
            self.exit_manager()
        except:
            pass
            
        self.openScreenName('setup')
        self.root.ids.setup_image.source = path
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def show_dialog_eddit(self):

        if self.dialog:
            self.dialog.dismiss()

        #if not self.dialog2:
        self.dialog = MDDialog(
                        title="Eddit project?",
                        text="Make a new project with same image.",
                        buttons=[
                        MDFlatButton(
                            text="CANCEL", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                        ),
                        MDFlatButton(
                            text="ACCEPT", text_color=self.theme_cls.primary_color, on_press=lambda x:self.eddit_project()
                        ),
                        ]
                        )
        self.dialog.open()

    def eddit_project(self):

        #self.img_path = os.path.dirname(self.root.ids.image.source)
        img_path = self.img_path+'/original.jpeg'
        self.openScreenName('setup')
        if self.dialog: self.dialog.dismiss()
        self.root.ids.setup_image.source = img_path
        toast(img_path)

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def goto_table(self):

        #self.openScreenName('mosaic_details')
        self.data_tables.open(self.root.ids.details)

    def back_main(self):

        self.data_tables.dismiss()
        self.openScreenName('main')

    def show_dialog_buy(self):

        if self.dialog:
            self.dialog.dismiss()

        #if not self.dialog2:
        self.dialog = MDDialog(
                        title="Cooming Soon",
                        text="Apologies, SmartBricks does not deliver yet. We are working to deliver your favourite mosaic to you. Are you interested in buying? ",
                        buttons=[
                        MDFlatButton(
                            text="NO", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                        ),
                        MDFlatButton(
                            text="YES", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                        ),
                        ]
                        )
        self.dialog.open()

    def show_dialog_empty(self):

        if self.dialog:
            self.dialog.dismiss()

        #if not self.dialog2:
        self.dialog = MDDialog(
                        title="Cooming Soon",
                        text="This option is not yet available.",
                        buttons=[
                        MDFlatButton(
                            text="Cancel", text_color=self.theme_cls.primary_color, on_press=lambda x:self.dialog.dismiss()
                        )
                        ]
                        )
        self.dialog.open()



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

self.data_tables = MDDataTable(
            size_hint=(0.9, 0.6),
            use_pagination=True,
            check=True,
            column_data=[
                ("No.", dp(30)),
                ("Column 1", dp(30)),
                ("Column 2", dp(30)),
                ("Column 3", dp(30)),
                ("Column 4", dp(30)),
                ("Column 5", dp(30)),
            ],
            row_data=[
                (f"{i + 1}", "2.23", "3.65", "44.1", "0.45", "62.5")
                for i in range(50)
            ],
        )
'''


if __name__ == "__main__":
    SmartBricksApp().run()
