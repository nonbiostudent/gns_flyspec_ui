# Copyright (C) Nial Peters 2015
#
# This file is part of gns_flyspec.
#
# gns_flyspec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gns_flyspec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.
import json
import threading
import multiprocessing
import numpy
import os
from PIL import Image
import time
import wx
import wx.calendar
import Queue
import datetime
import collections
import sys

from spectroscopy.flux import scans
from spectroscopy.flux import wind
from spectroscopy.flux import configuration
import matplotlib
matplotlib.use('wxagg')
import matplotlib.pyplot as plt
from matplotlib import dates

def __run_func(func, q, l, args, kwargs):
    """
    To make parallel_process compatible with Windows, the function passed to
    the Process constructor must be pickleable. It cannot therefore be a
    lambda function and so __run_func is defined instead.
    """
    q.put([func(i, *args, **kwargs) for i in l])

############################################################################

def parallel_process(func, list_, *args, **kwargs):
    """
    Runs the function 'func' on all items in list_ passing any additional
    args or kwargs specified. The list elements are processed asyncronously
    by as many processors as there are cpus. The return value will be a list
    of the return values of the function in the same order as the input list.
    """
    if len(list_) == 0:
        return []

    results = []
    processes = []
    queues = []

    for l in split(list_, multiprocessing.cpu_count()):
        q = multiprocessing.Queue(0)
        p = multiprocessing.Process(target=__run_func, args=(func, q, l, args, kwargs))
        p.start()
        processes.append(p)
        queues.append(q)

    for i in range(len(processes)):
        results.append(queues[i].get())
        processes[i].join()

    return flatten(results, ltypes=(list,))

############################################################################

def split(l, n):
    """
    splits the list l into n approximately equal length lists and returns
    them in a tuple. if n > len(l) then the returned tuple may contain
    less than n elements.
    >>> split([1,2,3],2)
    ([1, 2], [3])
    >>> split([1,2],3)
    ([1], [2])
    """
    length = len(l)

    # can't split into more pieces than there are elements!
    if n > length:
        n = length

    if int(n) <= 0:
        raise ValueError, "n must be a positive integer"

    inc = int(float(length) / float(n) + 0.5)
    split_list = []

    for i in range(0, n - 1):
        split_list.append(l[i * inc:i * inc + inc])

    split_list.append(l[n * inc - inc:])

    return tuple(split_list)

############################################################################

def flatten(l, ltypes=(list, tuple)):
    """
    Reduces any iterable containing other iterables into a single list
    of non-iterable items. The ltypes option allows control over what
    element types will be flattened. This algorithm is taken from:
    http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html

    >>> print flatten([range(3),range(3,6)])
    [0, 1, 2, 3, 4, 5]
    >>> print flatten([1,2,(3,4)])
    [1, 2, 3, 4]
    >>> print flatten([1,[2,3,[4,5,[6,[7,8,[9,[10]]]]]]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> print flatten([1,[2,3,[4,5,[6,[7,8,[9,[10]]]]]]], ltypes=())
    [1, [2, 3, [4, 5, [6, [7, 8, [9, [10]]]]]]]
    >>> print flatten([1,2,(3,4)],ltypes=(list))
    [1, 2, (3, 4)]
    """
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

############################################################################
def get_flux(s):
    return s.get_flux()


class ScanSummaryFrame:
    def __init__(self, config, realtime, liveview, from_png_file=None, day_to_process=None):
        self.__pending_callafter_finish = threading.Event()
        self.__update_required = threading.Event()
        self.__pending_callafter_finish.set()
        is_realtime = realtime
        show_liveview = liveview
        self.update_interval = 1
        self.__stay_alive = True
        self.config = config
        self.__fluxes = [None, None]
        self.__times = [None, None]
        self.__pickable_lines = [None, None]
        self.__static_lines = [None, None]
        self.__scans = [collections.deque([]), collections.deque([])]
        self.__scans_to_be_appended = [Queue.Queue(), Queue.Queue()]
        self.__scan_plot_styles = ['b-', 'g-']



        if from_png_file is not None:
            self.__load_png_file(from_png_file)
            self.__create_plots()
            self.__draw_plots()
            plt.show()
            self.__stay_alive = False
            return

        elif day_to_process is not None:
            if is_realtime:
                raise RuntimeError("Cannot process data for a specific day in realtime!")
            self.wind_data = wind.WindData(config, is_realtime, day_to_process=day_to_process)

            print "Looking for scan files matching: %s" % day_to_process.strftime("*%Y_%m_%d.txt")
            self.__scan_iters = [
                                 scans.ScanIter(self.wind_data, config["scanner1"], config,
                                                config["scanner1"]["data_location"],
                                                sort_func=cmp, realtime=is_realtime,
                                                skip_existing=False, pattern=day_to_process.strftime("*%Y_%m_%d*.txt")),
                                 scans.ScanIter(self.wind_data, config["scanner2"], config,
                                                config["scanner2"]["data_location"],
                                                sort_func=cmp, realtime=is_realtime,
                                                skip_existing=False, pattern=day_to_process.strftime("*%Y_%m_%d*.txt")),
                                 ]
        else:

            self.wind_data = wind.WindData(config, is_realtime)

            self.__scan_iters = [
                                 scans.ScanIter(self.wind_data, config["scanner1"], config,
                                                config["scanner1"]["data_location"],
                                                sort_func=cmp, realtime=is_realtime,
                                                skip_existing=True),
                                 scans.ScanIter(self.wind_data, config["scanner2"], config,
                                                config["scanner2"]["data_location"],
                                                sort_func=cmp, realtime=is_realtime,
                                                skip_existing=True),
                                 ]

        if show_liveview:
            self.__create_plots()

        if is_realtime:
            self.__worker_threads = [
                                     threading.Thread(target=self.__load_realtime, args=(0,)),
                                     threading.Thread(target=self.__load_realtime, args=(1,)),
                                     threading.Thread(target=self._update_plot)
                                     ]

            self.__worker_threads[0].start()
            self.__worker_threads[1].start()
            self.__worker_threads[2].start()
            if show_liveview:
                plt.show()
                self.close()

        else:
            self.__load_all_parallel()
            self.create_daily_plot()
            self.__draw_plots()
            if show_liveview:

                plt.show()

            self.close()


    def close(self):

        self.__stay_alive = False
        self.request_update()  # unblock the update thread

        for i in self.__scan_iters:
            i.close()

        if self.wind_data is not None:
            self.wind_data.close()


    def select_scan(self, scanner_idx, scan_idx):
        if not self.__scans[scanner_idx]:
            return

        scan = self.__scans[scanner_idx][scan_idx]
        self.__scan_plots[scanner_idx].clear()
        scan.plot_bkgd_fit(self.__scan_plots[scanner_idx], self.__scan_plot_styles[scanner_idx])
        self.__scan_plots[scanner_idx].relim()
        if scan._out_of_scan_range:
            self.__scan_plots[scanner_idx].set_title("Plume was out of scan range")
        elif scan.is_saturated:
            self.__scan_plots[scanner_idx].set_title("Scan contains saturated spectra")
        else:
            self.__scan_plots[scanner_idx].set_title("Plume angle guess = %d degrees" % int(scan._plume_pos_guess))

        self.__selection_pts[scanner_idx].set_data(self.__times[scanner_idx][scan_idx], self.__fluxes[scanner_idx][scan_idx])

        # select the point from the other scanner that is closest to this one
        t = self.__times[scanner_idx][scan_idx]

        other_scanner_idx = not scanner_idx  # this will break if there are more than two scanners
        if len(self.__times[other_scanner_idx]) == 0:
            self.fig.canvas.draw()
            return
        other_scan_idx = numpy.argmin(numpy.abs(self.__times[other_scanner_idx] - t))
        other_scan = self.__scans[other_scanner_idx][other_scan_idx]
        self.__scan_plots[other_scanner_idx].clear()
        other_scan.plot_bkgd_fit(self.__scan_plots[other_scanner_idx], self.__scan_plot_styles[other_scanner_idx])

        if other_scan._out_of_scan_range:
            self.__scan_plots[other_scanner_idx].set_title("Plume was out of scan range")
        elif other_scan.is_saturated:
            self.__scan_plots[other_scanner_idx].set_title("Scan contains saturated spectra")
        else:
            self.__scan_plots[other_scanner_idx].set_title("Plume angle guess = %d degrees" % int(other_scan._plume_pos_guess))

        self.__selection_pts[other_scanner_idx].set_data(self.__times[other_scanner_idx][other_scan_idx], self.__fluxes[other_scanner_idx][other_scan_idx])
        self.__scan_plots[other_scanner_idx].relim()

        self.fig.canvas.draw()


    def on_pick(self, evnt):
        if len(evnt.ind) > 1:
            scan_idx = evnt.ind[len(evnt.ind) // 2]
        else:
            scan_idx = evnt.ind

        scanner_idx = self.__pickable_lines.index(evnt.artist)
        self.select_scan(scanner_idx, scan_idx)


    def __create_plots(self):
        self.fig = plt.figure()

        self.__scan_plots = [
                             plt.subplot2grid((2, 2), (0, 0)),
                             plt.subplot2grid((2, 2), (0, 1))
                             ]
        self.__flux_plot = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)


    def __draw_plots(self):
        if len(self.__times[0]) == 0 and len(self.__times[1]) == 0:
            return
        self.__static_lines[0] = self.__flux_plot.plot(self.__times[0], self.__fluxes[0], 'b-')[0]
        self.__pickable_lines[0] = self.__flux_plot.plot(self.__times[0], self.__fluxes[0], 'b+', picker=5)[0]

        self.__static_lines[1] = self.__flux_plot.plot(self.__times[1], self.__fluxes[1], 'g-')[0]
        self.__pickable_lines[1] = self.__flux_plot.plot(self.__times[1], self.__fluxes[1], 'g+', picker=5)[0]

        self.__selection_pts = [
                                self.__flux_plot.plot([], [], 'r.', markersize=10)[0],
                                self.__flux_plot.plot([], [], 'r.', markersize=10)[0],
                                ]

        self.__flux_plot.axes.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        self.__flux_plot.set_xlabel("Time (local time)")
        self.__flux_plot.set_ylabel("SO$_2$ flux (kg/s)")
        self.select_scan(0, 0)
        self.select_scan(1, 0)

    def __load_all_parallel(self):
        print "loading scan data"


        self.__scans[0] = [i for i in self.__scan_iters[0]]
        self.__times[0] = numpy.array([i.times[0] for i in self.__scans[0]])

        self.__scans[1] = [i for i in self.__scan_iters[1]]
        self.__times[1] = numpy.array([i.times[0] for i in self.__scans[1]])

        all_fluxes = parallel_process(get_flux, self.__scans[0] + self.__scans[1])

        self.__fluxes[0] = all_fluxes[:len(self.__times[0])]
        self.__fluxes[1] = all_fluxes[len(self.__times[0]):]
        print "Loaded %d scans" % (len(self.__scans[0]) + len(self.__scans[1]))

        if len(self.__fluxes[0]) == 0 and len(self.__fluxes[1]) == 0:
            wx.MessageBox("Error! No scan data found for the requested dates.")
            self.close()
            wx.Yield()
            sys.exit(0)


    def __load_realtime(self, scanner_idx):
        for scan in self.__scan_iters[scanner_idx]:
            self.__scans_to_be_appended[scanner_idx].put(scan)
            self.request_update()


    def request_update(self):
        self.__update_required.set()


    def _update_plot(self):
        while self.__stay_alive:
            if self.__pending_callafter_finish.is_set():
                self.__pending_callafter_finish.clear()
                wx.CallAfter(self.__update)
            time.sleep(self.update_interval)

            self.__update_required.wait()
            self.__update_required.clear()

        # don't bother waiting for pending CallAfter calls to finish - just let
        # them fail (if we get to here then the plot is being deleted anyway)


    def __update(self):
        try:
            if self.__stay_alive:
                one_day = datetime.timedelta(days=1)
                new_day_flag = False
                tmp_storage = [[], []]
                for idx in [0, 1]:
                    while(True):
                        try:
                            s = self.__scans_to_be_appended[idx].get_nowait()
                            # check if it is a new day and therefore time to
                            # make the daily plot
                            if (self.__scans[idx] and
                                s.times[0].day != self.__scans[idx][-1].times[0].day):
                                new_day_flag = True
                                tmp_storage[idx].append(s)
                                break

                            self.__scans[idx].append(s)
                        except Queue.Empty:
                            break

                if new_day_flag:
                    self.__times[0] = numpy.array([i.times[0] for i in self.__scans[0]])
                    self.__times[1] = numpy.array([i.times[0] for i in self.__scans[1]])

                    self.__fluxes[0] = [i.get_flux() for i in self.__scans[0]]
                    self.__fluxes[1] = [i.get_flux() for i in self.__scans[1]]
                    self.create_daily_plot()
                    for idx in [0, 1]:
                        for s in tmp_storage[idx]:
                            self.__scans[idx].append(s)
                    wx.CallAfter(self.request_update)

                for idx in [0, 1]:
                    # now crop the list of scans to be <=1day in length
                    while len(self.__scans[idx]) > 1:
                        t_range = self.__scans[idx][-1].times[0] - self.__scans[idx][0].times[0]
                        if t_range > one_day:
                            self.__scans[idx].popleft()  # remove the first item
                        else:
                            break

                self.__times[0] = numpy.array([i.times[0] for i in self.__scans[0]])
                self.__times[1] = numpy.array([i.times[0] for i in self.__scans[1]])

                self.__fluxes[0] = [i.get_flux() for i in self.__scans[0]]
                self.__fluxes[1] = [i.get_flux() for i in self.__scans[1]]

                if self.__static_lines[0] is None:
                    self.__draw_plots()

                # now update the plots
                if self.__static_lines[0] is not None:
                    self.__static_lines[0].set_data(self.__times[0], self.__fluxes[0])
                    self.__pickable_lines[0].set_data(self.__times[0], self.__fluxes[0])
                if self.__static_lines[1] is not None:
                    self.__static_lines[1].set_data(self.__times[1], self.__fluxes[1])
                    self.__pickable_lines[1].set_data(self.__times[1], self.__fluxes[1])

                self.__flux_plot.relim()
                self.__flux_plot.autoscale_view()
                self.select_scan(0, len(self.__scans[0]) - 1)



        finally:
            self.__pending_callafter_finish.set()


    def create_daily_plot(self):
        # get a representative time for the data
        if len(self.__times[0]) > 0:
            rep_time = self.__times[0][0]
        else:
            rep_time = self.__times[1][0]

        # write the config data into the png header
        fig = plt.figure()

        plt.plot(self.__times[0], self.__fluxes[0], 'b-', label=self.config["scanner1"]["name"])
        plt.plot(self.__times[0], self.__fluxes[0], 'b+')

        plt.plot(self.__times[1], self.__fluxes[1], 'g-', label=self.config["scanner2"]["name"])
        plt.plot(self.__times[1], self.__fluxes[1], 'g+')

        plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))

        plt.xlabel("Time")
        plt.ylabel("SO$_2$ Flux (kg/s)")
        plt.title(rep_time.strftime("%d %b %Y"))
        plt.legend()


        output_filename = os.path.join(self.config["daily_plots_folder"],
                                       rep_time.strftime("%Y"),
                                       rep_time.strftime("%b"),
                                       rep_time.strftime("%Y_%m_%d.png"))

        if len(self.__times[0]) > 0:
            output_txtfile1 = os.path.join(self.config["daily_plots_folder"],
                                           rep_time.strftime("%Y"),
                                           rep_time.strftime("%b"),
                                           rep_time.strftime(self.config["scanner1"]["name"] + "_%Y_%m_%d.txt"))

        if len(self.__times[1]) > 0:
            output_txtfile2 = os.path.join(self.config["daily_plots_folder"],
                                           rep_time.strftime("%Y"),
                                           rep_time.strftime("%b"),
                                           rep_time.strftime(self.config["scanner2"]["name"] + "_%Y_%m_%d.txt"))

        # create any needed sub-folders
        try:
            os.makedirs(os.path.dirname(output_filename))
        except OSError:
            # dir exists
            pass

        if len(self.__times[0]) > 0:
            with open(output_txtfile1, "w") as ofp:
                ofp.write("#\tTime\t\t\tSO2 Flux (kg/s)\n")
                for i in range(len(self.__times[0])):
                    ofp.write("%s\t\t%f\n" % (self.__times[0][i], self.__fluxes[0][i]))

        if len(self.__times[1]) > 0:
            with open(output_txtfile2, "w") as ofp:
                ofp.write("#\tTime\t\t\tSO2 Flux (kg/s)\n")
                for i in range(len(self.__times[1])):
                    ofp.write("%s\t\t%f\n" % (self.__times[1][i], self.__fluxes[1][i]))





        plt.savefig(output_filename, format='png')
        plt.close(fig)
        # now convert this to PIL image
        im = Image.open(output_filename)

        # store the scan data in the image header
        im.info["config"] = json.dumps(self.config)
        im.info["scanner1_data"] = "[" + ','.join([s.toJSON() for s in self.__scans[0]]) + "]"
        im.info["scanner2_data"] = "[" + ','.join([s.toJSON() for s in self.__scans[1]]) + "]"

        # use pngsave function since im.save does not preserve header data
        pngsave(im, output_filename)


    def __load_png_file(self, filename):

        im = Image.open(filename)
        self.config = json.loads(im.info["config"])

        self.__scans[0] = collections.deque([scans.Scan.fromJSON(s) for s in json.loads(im.info["scanner1_data"])])
        self.__times[0] = numpy.array([i.times[0] for i in self.__scans[0]])

        self.__scans[1] = collections.deque([scans.Scan.fromJSON(s) for s in json.loads(im.info["scanner2_data"])])
        self.__times[1] = numpy.array([i.times[0] for i in self.__scans[1]])

        self.__fluxes[0] = [s.get_flux() for s in self.__scans[0]]
        self.__fluxes[1] = [s.get_flux() for s in self.__scans[1]]


def pngsave(im, file_):
    """
    Function saves a PIL image as a PNG file, preserving the header data
    """

    # these can be automatically added to Image.info dict
    # they are not user-added metadata
    reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')

    # undocumented class
    from PIL import PngImagePlugin
    meta = PngImagePlugin.PngInfo()

    # copy metadata into new object
    for k, v in im.info.iteritems():
        if k in reserved: continue
        meta.add_text(k, v, 0)

    # and save
    im.save(file_, "PNG", pnginfo=meta)



def main():
    """
    This is the function that gets called when you run the flyspec-realtime program
    """
    config = configuration.load_config()

    ScanSummaryFrame(config, True, True)

def wxdate2pydate(date):
    assert isinstance(date, wx.DateTime)
    if date.IsValid():
        ymd = map(int, date.FormatISODate().split('-'))
        return datetime.date(*ymd)
    else:
        return None

class DayToProcessPickerDialog(wx.Dialog):
    def __init__(self):
        wx.Dialog.__init__(self, None, -1, "GNS Flyspec")

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.cal = wx.calendar.CalendarCtrl(self, -1)

        vsizer.Add(wx.StaticText(self, -1, "Choose day to process"), 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, border=10)

        vsizer.Add(self.cal, 1, wx.EXPAND | wx.ALL, border=10)

        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.ok_button = wx.Button(self, wx.ID_ANY, "OK")
        self.cancel_button = wx.Button(self, wx.ID_ANY, "Cancel")
        buttons_sizer.Add(self.ok_button, 0, wx.ALIGN_RIGHT)
        buttons_sizer.Add(self.cancel_button, 0, wx.ALIGN_RIGHT)

        vsizer.Add(buttons_sizer, 0, wx.ALIGN_RIGHT | wx.ALIGN_BOTTOM | wx.ALL, border=10)

        wx.EVT_BUTTON(self, self.cancel_button.GetId(), self.on_cancel)
        wx.EVT_BUTTON(self, self.ok_button.GetId(), self.on_ok)
        self.selected_date = None

        self.SetAutoLayout(1)
        self.SetSizer(vsizer)
        vsizer.Fit(self)

    def on_ok(self, evnt):
        self.EndModal(wx.OK)
        self.selected_date = wxdate2pydate(self.cal.GetDate())
        self.Destroy()
    def on_cancel(self, evnt):
        self.EndModal(wx.CANCEL)
        self.Destroy()




def main_no_realtime():
    app = wx.PySimpleApp()

    day_picker = DayToProcessPickerDialog()

    if day_picker.ShowModal() == wx.OK:
        print "Processing data for %s" % day_picker.selected_date
        config = configuration.load_config()
        ScanSummaryFrame(config, False, True, day_to_process=day_picker.selected_date)
    app.MainLoop()


