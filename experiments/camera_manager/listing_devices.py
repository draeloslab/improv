""" List all tis-camera devices available on the system """
import gi
import sys

gi.require_version("Tcam","1.0")
gi.require_version("Gst","1.0")

from gi.repository import Tcam, Gst

Gst.init(sys.argv)

monitor = Gst.DeviceMonitor.new()
monitor.add_filter("Video/Source/tcam")

# listing devices
devices = []
           
for device in monitor.get_devices():
    struc = device.get_properties()

    print("\tmodel:\t{}\tserial:\t{}\ttype:\t{}".format(struc.get_string("model"),
                                                        struc.get_string("serial"),
                                                        struc.get_string("type")))