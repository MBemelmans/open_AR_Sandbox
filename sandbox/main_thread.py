import asyncio

import collections
import numpy
import threading
import panel as pn
pn.extension()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import traceback
from datetime import datetime
import skimage.transform
import platform
_platform = platform.system()

dateTimeObj = datetime.now()

from sandbox.projector import Projector, ContourLinesModule, CmapModule
from sandbox.sensor import Sensor
from sandbox.markers import MarkerDetection
from sandbox import set_logger
logger = set_logger(__name__)


class MainThread:
    """
    Module with threading methods
    """

    def __init__(self, sensor: Sensor, projector: Projector, aruco: MarkerDetection = None,
                 check_change: bool = False, kwargs_contourlines: dict = {}, kwargs_cmap: dict = {},
                 **kwargs):
        """

        Args:
            sensor:
            projector:
            aruco:
            modules:
            crop:
            clip:
            check_change:
            **kwargs:
        """
        self._error_message = ''
        # TODO: in all the modules be carefull with zorder
        self.sensor = sensor
        self.projector = projector
        self.projector.clear_axes()
        self.contours = ContourLinesModule(extent=self.sensor.extent, **kwargs_contourlines)
        self.cmap_frame = CmapModule(extent=self.sensor.extent, **kwargs_cmap)

        # start the modules
        self.modules = collections.OrderedDict({'CmapModule': self.cmap_frame, 'ContourLinesModule': self.contours})
        self._modules = collections.OrderedDict({'CmapModule': self.cmap_frame, 'ContourLinesModule': self.contours})

        # threading
        self.lock = threading.Lock()
        self.thread = None
        self.thread_status = 'stopped'  # status: 'stopped', 'running', 'paused'
        self.main_task = None

        # connect to ArucoMarker class
        # if CV2_IMPORT is True:
        self.Aruco = aruco
        self.ARUCO_ACTIVE = False
        if isinstance(self.Aruco, MarkerDetection):
            self.ARUCO_ACTIVE = True

        self.sb_params = {'frame': self.sensor.get_frame(),
                          'ax': self.projector.ax,
                          'set_colorbar': self.projector.set_colorbar,
                          'set_legend': self.projector.set_legend,
                          'extent': self.sensor.extent,
                          'box_dimensions': self.sensor.physical_dimensions,
                          'marker': pd.DataFrame(),
                          'cmap': plt.cm.get_cmap('gist_earth'),
                          'norm': None,
                          'active_cmap': True,
                          'active_shading': True,
                          'active_contours': True,
                          'same_frame': False,
                          'lock_thread': self.lock,
                          'trigger': self.projector.trigger,
                          # TODO: Carefull with this use because it can make to paint the figure incompletely
                          'del_contour': True, }
        # 'freeze_frame': False}

        self.previous_frame = self.sb_params['frame']

        # To reduce the noise of the data
        self.check_change = check_change
        self._rtol = 0.07  # Widgets for this
        self._atol = 0.001
        # render the frame
        self.cmap_frame.render_frame(self.sb_params['frame'], self.sb_params['ax'])
        # plot the contour lines
        self.contours.plot_contour_lines(self.sb_params['frame'], self.sb_params['ax'])
        self.projector.trigger()

        self._create_widgets()

        self._loaded_frame = False
        self._error_message = ''
        self._widget_error_markdown = pn.pane.Markdown("<p>Open_AR_Sandbox</p>")

        # <MB>
        self.saved_frame = None  # Initialize saved_frame
        self.InSAR_mode_flag = False # indicate InSAR mode on (1) or off (0)
        self.diff_mod = 10 # default difference modulator for InSAR mode
        self.diff_minmax = 10 # min max of total height diff for unwrapped insar mode

        # initialise dismph cmap
        from matplotlib.colors import LinearSegmentedColormap
        clist = ['#f579cd', '#f67fc6', '#f686bf', '#f68cb9', '#f692b3', '#f698ad',
                            '#f69ea7', '#f6a5a1', '#f6ab9a', '#f6b194', '#f6b78e', '#f6bd88',
                            '#f6c482', '#f6ca7b', '#f6d075', '#f6d66f', '#f6dc69', '#f6e363',
                            '#efe765', '#e5eb6b', '#dbf071', '#d0f477', '#c8f67d', '#c2f684',
                            '#bbf68a', '#b5f690', '#aff696', '#a9f69c', '#a3f6a3', '#9cf6a9',
                            '#96f6af', '#90f6b5', '#8af6bb', '#84f6c2', '#7df6c8', '#77f6ce',
                            '#71f6d4', '#6bf6da', '#65f6e0', '#5ef6e7', '#58f0ed', '#52e8f3',
                            '#4cdbf9', '#7bccf6', '#82c4f6', '#88bdf6', '#8eb7f6', '#94b1f6',
                            '#9aabf6', '#a1a5f6', '#a79ef6', '#ad98f6', '#b392f6', '#b98cf6',
                            '#bf86f6', '#c67ff6', '#cc79f6', '#d273f6', '#d86df6', '#de67f6',
                            '#e561f6', '#e967ec', '#ed6de2', '#f173d7']
        colormap = LinearSegmentedColormap.from_list('dismph', clist, N=256)
        if cm.get_cmap('dismph') is None:
            from matplotlib.cm import register_cmap
            register_cmap(name ='dismph', cmap=colormap)
        # </MB>

    # @property @TODO: test if this works
    # def sb_params(self):
    #    return {'frame': self.sensor.get_frame(),
    #              'ax': self.projector.ax,
    #              'extent': self.sensor.extent,
    #              'marker': pd.DataFrame(),
    #              'cmap': plt.cm.get_cmap('gist_earth'),
    #              'norm': None,
    #              'active_cmap': True,
    #              'active_contours': True,
    #              'same_frame': False,
    #              'freeze_frame': False}

    def update(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:

        """
        self.sb_params['ax'] = self.projector.ax

        if self._loaded_frame:
            frame = self.previous_frame  # if loaded DEM the previous frame will have this information
            self.sb_params['extent'] = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]
            self.sb_params[
                'same_frame'] = False  # TODO: need to organize the usage of same_frame because is contradictory
        else:
            frame = self.sensor.get_frame()
            self.sb_params['extent'] = self.sensor.extent
            # This is to avoid noise in the data
            if self.check_change:
                cl = numpy.isclose(self.previous_frame, frame, atol=self._atol, rtol=self._rtol, equal_nan=True)
                self.previous_frame[numpy.logical_not(cl)] = frame[numpy.logical_not(cl)]
                frame = self.previous_frame
                # self.sb_params['same_frame'] = True #TODO: Check for usage of this part
            else:
                self.previous_frame = frame
                self.sb_params['same_frame'] = False

        if self.saved_frame is not None and self.InSAR_mode_flag == True:
            current_frame = self.sensor.get_frame()
            difference = current_frame - self.saved_frame
            if self.wrapped == True:
                frame = numpy.mod(difference,self.diff_mod)
                cmap_bounds = [ frame.min(), frame.max()]
            else:
                frame = difference
                cmap_bounds = [-self.diff_minmax, self.diff_minmax]

            self.sb_params['extent'] = [0, frame.shape[1], 0, frame.shape[0], cmap_bounds[0], cmap_bounds[1]]
        
        self.sb_params['frame'] = frame

        # filter
        self.lock.acquire()
        if self.ARUCO_ACTIVE:
            df = self.Aruco.update()
        else:
            df = pd.DataFrame()
        self.lock.release()

        self.sb_params['marker'] = df

        try:
            if self._error_message:
                self.sb_params['ax'].texts = []
                self._error_message = ''
                self._widget_error_markdown.object = "Running"
            self.lock.acquire()
            _cmap = ['CmapModule'] if 'CmapModule' in self.modules.keys() else []
            _contours = ['ContourLinesModule'] if 'ContourLinesModule' in self.modules.keys() else []
            _always = _cmap + _contours
            _actual = [name for name in self.modules.keys() if name not in _always]
            for key in list(
                    _actual + _always):  # TODO: maybe use OrderedDict to put this modules always at the end of the iteration
                self.sb_params = self.modules[key].update(self.sb_params)
            self.lock.release()
        except Exception as e:
            traceback.print_exc()
            logger.critical(e, exc_info=True)
            self._error_message = str(dateTimeObj) + str(type(e)) + str(e)
            self._widget_error_markdown.object = self._error_message
            self.lock.release()
            self.thread_status = 'stopped'
            self.projector.write_text("Ups... Something went wrong. The Thread is paused..."
                                      "\n Check 'self._error_message' to see what happened"
                                      "\n or open the 'sandbox.log' file for a detailed description")

        self.sb_params['ax'].set_xlim(xmin=self.sb_params.get('extent')[0], xmax=self.sb_params.get('extent')[1])
        self.sb_params['ax'].set_ylim(ymin=self.sb_params.get('extent')[2], ymax=self.sb_params.get('extent')[3])

        if isinstance(self.Aruco, MarkerDetection):
            _ = self.Aruco.plot_aruco(self.sb_params['ax'], self.sb_params['marker'])
            # Update of legend
            self.sb_params['set_legend'](self.Aruco.legend_elements)
        self.lock.acquire()
        self.projector.trigger()
        self.lock.release()

    def load_frame(self, frame: numpy.ndarray = None, from_file: str=None):
        """
        During the sandbox thread, if you want to fix a frame but not interrupt the thread,
        load the desired numpy array here.
        This will change the flag self._loaded_frame = False to True in the update function and
        stop the sensor frame acquisition.
        # TODO: Make this compatible with DEM of differ shapes (e.g. bennisson DEM)
        To stop this, pass: frame = None or change the flag self._loaded_frame to False.
        Args:
            frame: numpy.ndarray: must be a matrix of desired resolution
            from_file: Path to DEM. Can be either .npz or .npy (internally normalized)
        Returns:

        """
        if frame is None and from_file is None:
            self._loaded_frame = False
            logger.info("No frame to load, resuming sandbox frame acquisition")
            return False

        if isinstance(from_file, str):
            try:
                file = numpy.load(from_file)
            except Exception:
                logger.error("%s as path not valid" % from_file, exc_info=True)
                return False
            if from_file.split(".")[1] == "npy":
                frame_new = self.normalize_topography(file, self.sensor.extent)
                self._loaded_frame = True
                self.previous_frame = frame_new
                logger.info("loaded .npy file")
                return True
            elif from_file.split(".")[1] == "npz":
                frame_new = file["arr_0"]
                frame_new = frame_new - frame_new.min()
                self._loaded_frame = True
                self.previous_frame = frame_new
                logger.info("loaded .npz file")
                return True
            else:
                logger.error("%s format not recognized. Please pass a .npy or .npz file" % from_file)
                return False

        if isinstance(frame, numpy.ndarray):
            self._loaded_frame = True
            self.previous_frame = frame
            logger.info("loaded")
            return True
        logger.error("Frame and path not valid: Frame -> %s, Path -> %s" % (frame, from_file))
        return False

    def save_frame(self, to_file: str):
        """
        Save the current frame to a .npy or .npz file.

        Args:
            to_file (str): Path to the file where the frame should be saved.
        """
        if self.previous_frame is None:
            logger.error("No frame to save")
            return False

        ext = to_file.split(".")[-1]
        try:
            if ext == "npy":
                numpy.save(to_file, self.previous_frame)
                logger.info(f"Frame saved to {to_file}")
                return True
            elif ext == "npz":
                numpy.savez_compressed(to_file, self.previous_frame)
                logger.info(f"Frame saved to {to_file}")
                return True
            else:
                logger.error(f"{to_file} format not recognized. Please use .npy or .npz")
                return False
        except Exception as e:
            logger.error(f"Failed to save frame: {e}", exc_info=True)
            return False

    def change_colormap(self, cmap_name: str):
        """
        Change the colormap of the current frame.

        Args:
            cmap_name (str): Name of the colormap to use.
        """
        try:
            cmap = cm.get_cmap(cmap_name)
            self.sb_params['cmap'] = cmap
            self.cmap_frame.update(self.sb_params)
            logger.info(f"Colormap changed to {cmap_name}")
        except Exception as e:
            logger.error(f"Failed to change colormap: {e}", exc_info=True)
            return False
        return True
    

    def save_frame2(self):
        self.saved_frame = self.previous_frame.copy()
        
        logger.info(f"Current frame saved for InSAR mode. size {numpy.shape(self.saved_frame)}")

    def unload_frame(self):
        self.saved_frame = None
        self.InSAR_mode_flag = False
        self.sb_params['cmap'] = cm.get_cmap('gist_earth')
        self.sb_params['active_shading'] = True # re-enable shading for topo map
        self.sb_params['active_contours'] = True
        logger.info("Removed save frame, return to default topography mode.")

    def InSAR_mode(self, wrapped = True, diff_mod = 10, cmap = 'dismph',contour_flag=False,diff_minmax = 10):
        # self.save_frame2()
        self.diff_mod = diff_mod
        self.InSAR_mode_flag = True
        self.wrapped = wrapped
        self.diff_minmax = diff_minmax
        self.sb_params['active_shading'] = False # Disable shading for difference map
        self.sb_params['active_contours'] = contour_flag
        self.sb_params['cmap'] = cm.get_cmap(cmap)

        self.thread_status = 'running'
        asyncio.create_task(self.thread_loop())


    def show_lahars(self, elev, lahars, stream = None, ec_line = None, ):
        """
        Display the lahar simulation results.

        Args:
            elev: numpy.ndarray - Elevation data.
            stream: numpy.ndarray - Stream flow lines (rasterized).
            ec_line: numpy.ndarray - Energy cone line (rasterized).
            lahars: numpy.ndarray - Lahar inundation area.
        """
        try:
            # Update elevation frame
            self.load_frame(frame=elev)
            self.update()

            # Create a colormap for the different layers
            stream_cmap = cm.get_cmap('Greys')
            ec_line_cmap = cm.get_cmap('Greens')
            lahars_cmap = cm.get_cmap('Reds')

            # make lahar_alpha
            lahar_alpha = numpy.zeros_like(lahars)
            lahar_alpha[lahars>0] = 1

            # ax.imshow(elev_data,cmap='viridis')

            # ax.imshow(stream_data,cmap='Greys',alpha=stream_data,zorder=100,interpolation='nearest')
            # # ax.imshow(stream_data,cmap='Greens')

            # ax.imshow(EC_line_data,cmap='Greens',alpha = EC_line_data,interpolation='nearest')

            # ax.imshow(lahar_data,cmap='Reds',alpha = lahar_data,interpolation='nearest')

            
            # Overlay the energy cone line
            if ec_line is not None:
                self.projector.ax.imshow(ec_line, cmap=ec_line_cmap, alpha=ec_line, extent=self.sensor.extent, zorder=1, interpolation='nearest')

            # Overlay the lahar inundation area
            self.projector.ax.imshow(lahars, cmap=lahars_cmap, alpha=lahar_alpha, extent=self.sensor.extent, zorder=2, interpolation='nearest')

            # Overlay the stream flow lines
            if stream is not None:
                self.projector.ax.imshow(stream, cmap=stream_cmap, alpha=stream, extent=self.sensor.extent, zorder=3, interpolation='nearest')
            
            # Trigger the projector to render the updated frame
            self.projector.trigger()

            logger.info("Lahar simulation results displayed successfully.")
        except Exception as e:
            logger.error(f"Failed to display lahar simulation results: {e}", exc_info=True)




    @staticmethod
    def normalize_topography(dem, target_extent):
        """
        # TODO: Multiple implementations. TopoModule and LoadSaveModule
        Normalize any size of numpy array to fit the sandbox frame.
        Useful when passing DEM with resolution bigger than sandbox sensor.
        Args:
            dem:
            target_extent: [minx, maxx, miny, maxy, vmin, vmax] ->
            [0, frame_width, 0, frame_height, vmin_sensor, vmax_sensor]
        Returns:
             normalized frame
        """
        # Change shape of numpy array to desired shape
        topo_changed = skimage.transform.resize(dem,
                                                (target_extent[3], target_extent[1]),
                                                order=3,
                                                mode='edge',
                                                anti_aliasing=True,
                                                preserve_range=False)

        topo_min = topo_changed.min()
        topo_max = topo_changed.max()
        # when the min value is not 0
        topo_changed = topo_changed - topo_min
        topo_changed = topo_changed * (target_extent[-1] - target_extent[-2]) / (topo_max - topo_min)

        return topo_changed

    def add_module(self, name: str, module):
        """Add an specific module to run the update in the main thread"""
        self.lock.acquire()
        self.modules[name] = module
        self._modules[name] = module
        self.lock.release()
        # self.modules.move_to_end(name, last=True)
        logger.info('module ' + name + ' added to modules')

    def remove_module(self, name: str):
        """Remove a current module from the main thread"""
        if name in self.modules.keys():
            self.lock.acquire()
            self.modules.pop(name)
            logger.info('module ' + name + ' removed')
            self.lock.release()
        else:
            logger.warning('No module with name ' + name + ' was found')

    def module_manager(self, active_modules: list = []):
        # add a module
        for name_module in active_modules:
            if name_module not in self.modules:
                self.add_module(name=name_module, module=self._modules[name_module])
        # delete the module
        if len(active_modules) > 0:
            [self.remove_module(name) for name in self.modules if name not in active_modules]
        # self._update_widget_module_selector()
        # if self._widget_module_selector is not None:
        #    self._widget_module_selector.value = list(self.modules.keys())
        #    self._widget_module_selector.options = list(self._modules.keys())

    def _update_widget_module_selector(self):
        if self._widget_module_selector is not None:
            self._widget_module_selector.value = list(self.modules.keys())
            self._widget_module_selector.options = list(self._modules.keys())


    async def thread_loop(self):
        while self.thread_status == 'running':
            self.update()
            await asyncio.sleep(0.1) #give other threads a chance to run

    def run(self):
        if self.thread_status != 'running':
            if _platform == "Linux":
                if self.sensor.s_name == "kinect_v2" or self.sensor.s_name == "lidar":
                    self.sensor.Sensor._run()
            self.thread_status = 'running'
            self.main_task = asyncio.create_task(self.thread_loop())
            #self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            #self.thread.start()
            logger.info('Thread started or resumed...')

        else:
            logger.info('Thread already running.')

    def stop(self):
        if self.thread_status is not 'stopped':
            if _platform == "Linux":
                if self.sensor.s_name == "kinect_v2" or self.sensor.s_name == "lidar":
                    self.sensor.Sensor._stop()
            self.thread_status = 'stopped'  # set flag to end thread loop
            #self.thread.join()  # wait for the thread to finish
            self.main_task.cancel()
            logger.info('Thread stopped.')
        else:
            logger.info('thread was not running.')

    def pause(self):
        if self.thread_status == 'running':
            self.thread_status = 'paused'  # set flag to end thread loop
            #self.thread.join()  # wait for the thread to finish
            self.main_task.cancel()
            logger.info('Thread paused.')
        else:
            logger.info('There is no thread running.')

    def resume(self):
        if self.thread_status != 'stopped':
            self.run()
        else:
            logger.info('Thread already stopped.')

    def widget_plot_module(self):
        if isinstance(self.Aruco, MarkerDetection):
            marker = pn.Column(self.widgets_aruco_visualization(), self.widget_thread_controller())
            widgets = pn.Column(self.cmap_frame.show_widgets(),
                                self.contours.show_widgets())
            rows = pn.Row(widgets, marker)
        else:
            widgets = pn.Column(self.cmap_frame.show_widgets(),
                                self.contours.show_widgets())
            rows = pn.Row(widgets, self.widget_thread_controller())

        panel1 = pn.Column("## Plotting interaction widgets", rows)
        self._update_widget_module_selector()

        panel2 = self.projector.show_widgets_sidepanels()

        panel = pn.Tabs(("Main frame controller", panel1),
                        ("Side panel controller", panel2))

        return panel

    def widget_thread_controller(self):
        panel = pn.Column("##<b>Thread Controller</b>",
                          self._widget_thread_selector,
                          self._widget_check_difference,
                          self._widget_module_selector,
                          self._widget_clear_axes,
                          self._widget_error_markdown
                          # self._widget_freeze_frame)
                          )
        return panel

    def _create_widgets(self):
        self._widget_thread_selector = pn.widgets.RadioButtonGroup(name='Thread controller',
                                                                   options=["Start", "Stop"],
                                                                   value="Start",
                                                                   button_type='success')
        self._widget_thread_selector.param.watch(self._callback_thread_selector, 'value', onlychanged=False)

        self._widget_check_difference = pn.widgets.Checkbox(name='Check changes in fame', value=self.check_change)
        self._widget_check_difference.param.watch(self._callback_check_difference, 'value',
                                                  onlychanged=False)

        self._widget_clear_axes = pn.widgets.Button(name="Clear axes from projector / refresh list",
                                                    button_type="warning")
        self._widget_clear_axes.param.watch(self._callback_clear_axes, 'clicks',
                                            onlychanged=False)

        self._widget_module_selector = pn.widgets.CrossSelector(name="Module manager",
                                                                value=list(self.modules.keys()),
                                                                options=list(self._modules.keys()),
                                                                definition_order=False)
        self._widget_module_selector.param.watch(self._callback_module_selector, 'value',
                                                 onlychanged=False)

    def _callback_clear_axes(self, event):
        self.projector.clear_axes()
        self._update_widget_module_selector()
        # if self._widget_module_selector is not None:
        #    self._widget_module_selector.value = list(self.modules.keys())
        #    self._widget_module_selector.options = list(self._modules.keys())

    def _callback_check_difference(self, event):
        self.check_change = event.new

    def _callback_module_selector(self, event):
        self.module_manager(event.new)

    def _callback_thread_selector(self, event):
        if event.new == "Start":
            self.run()
        elif event.new == "Stop":
            self.stop()

    def widgets_aruco_visualization(self):
        self._widget_aruco = pn.widgets.Checkbox(name='Aruco Detection', value=self.ARUCO_ACTIVE)
        self._widget_aruco.param.watch(self._callback_aruco, 'value',
                                       onlychanged=False)
        panel = pn.Column("## Activate aruco detection", self._widget_aruco, self.Aruco.widgets_aruco())
        return panel

    def _callback_aruco(self, event):
        self.ARUCO_ACTIVE = event.new
