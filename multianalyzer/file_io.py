__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "20/05/2022"

import os
import posixpath
import sys
import time
import numpy
import logging
logger = logging.getLogger(__name__)
try:
    import h5py
except ImportError as error:
    h5py = None
    logger.error("h5py module missing")
else:
    try:
        h5py._errors.silence_errors()
    except AttributeError:  # old h5py
        pass
try:
    import hdf5plugin
except ImportError:
    CMP = {"chunks":True,
           "compression": "gzip",
           "compression_opts":1}
else:
    CMP = hdf5plugin.Bitshuffle()

from . import version


def topas_parser(infile):
    """Parser for TOPAS output file with geometry of the multianalyzer
    
    :param infile: name of the file containing the TOPAS refined parameters, usually 'out7.pars'
    :return: dict with the data from the file (no conversion is made)
    """
    res = {}
    with open(infile, "r") as f:
        keys = ["centre", "rollx", "rolly", "offset"]
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == "Wavelength":
                res["wavelength"] = float(words[2])
            if words[:2] == ["manom", "mantth"]:
                lkeys = words
                line = f.readline()
                words = line.split()
                for k,v in zip(lkeys, words):
                    res[k] = float(v)
            if words[:2] == ["L1", "L2"]:
                line = f.readline()
                words = line.split()
                res["L1"] = float(words[0])
                res["L2"] = float(words[1])
            if len(words)>=len(keys) and words[:len(keys)] == keys:
                keys = words
                for k in keys:
                    res[k] = []
                for line in f:
                    words = line.split()
                    if len(words) < len(keys) + 1:
                        break
                    for k, v in zip(keys, words):
                        res[k].append(float(v))
    return res


def ID22_bliss_parser(infile, entry=None):
    """Read an HDF5 file as generated by BLISS on ID22 and returns valuable information from there
    
    :param infile: name of the HDF5 file
    :return: dict with entries containing each:
        * tha: the acceptance angle of the analyzer
        * thd: the diffractio angle of the analyzer (~2xtha)
        * roicol: A collection of ROI for each frame
        * mon: a monitor for the intensity of the incoming beam
        * arm: the position of the 2theta arm when collecting 
    """
    res = {}
    with h5py.File(infile, "r") as h:
        entries = [k for k, v in h.items() if v.attrs.get("NX_class") == "NXentry"]
        if entry and entry in entries:
            entry_dict = {}
            entry_grp = h[entry]
            entry_dict["roicol"] = entry_grp["measurement/eiger_roi_collection"][()]
            entry_dict["arm"] = entry_grp["measurement/tth"][()]
            entry_dict["mon"] = entry_grp["measurement/mon"][()]
            entry_dict["tha"] = entry_grp["instrument/positioners/manom"][()]
            entry_dict["thd"] = entry_grp["instrument/positioners/mantth"][()]
            res[entry] = entry_dict
        else:
            entries.sort()
            for entry in entries:
                entry_dict = {}
                if not entry.endswith(".1"):
                    continue
                title = entry+"/title"
                if title in h:
                    title = h[title][()]
                    try:
                        title = title.decode()
                    except:
                        pass
                    if title.startswith("fscan"):
                        entry_grp = h[entry]
                        entry_dict["roicol"] = entry_grp["measurement/eiger_roi_collection"][()]
                        entry_dict["arm"] = entry_grp["measurement/tth"][()]
                        entry_dict["mon"] = entry_grp["measurement/mon"][()]
                        entry_dict["tha"] = entry_grp["instrument/positioners/manom"][()]
                        entry_dict["thd"] = entry_grp["instrument/positioners/mantth"][()]
                        res[entry] = entry_dict
    return res


def get_isotime(forceTime=None):
    """
    :param forceTime: enforce a given time (current by default)
    :type forceTime: float
    :return: the current time as an ISO8601 string
    :rtype: string
    """
    if forceTime is None:
        forceTime = time.time()
    localtime = time.localtime(forceTime)
    gmtime = time.gmtime(forceTime)
    tz_h = localtime.tm_hour - gmtime.tm_hour
    tz_m = localtime.tm_min - gmtime.tm_min
    return "%s%+03i:%02i" % (time.strftime("%Y-%m-%dT%H:%M:%S", localtime), tz_h, tz_m)


def from_isotime(text, use_tz=False):
    """
    :param text: string representing the time is iso format
    """
    if len(text) == 1:
        # just in case someone sets as a list
        text = text[0]
    try:
        text = text.decode("ascii")
    except (UnicodeError, AttributeError):
        text = str(text)
    if len(text) < 19:
        logger.warning("Not a iso-time string: %s", text)
        return
    base = text[:19]
    if use_tz and len(text) == 25:
        sgn = 1 if text[:19] == "+" else -1
        tz = 60 * (60 * int(text[20:22]) + int(text[23:25])) * sgn
    else:
        tz = 0
    return time.mktime(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + tz


def is_hdf5(filename):
    """
    Check if a file is actually a HDF5 file

    :param filename: this file has better to exist
    """
    signature = [137, 72, 68, 70, 13, 10, 26, 10]
    if not os.path.exists(filename):
        raise IOError("No such file %s" % (filename))
    with open(filename, "rb") as f:
        raw = f.read(8)
    sig = [ord(i) for i in raw] if sys.version_info[0] < 3 else [int(i) for i in raw]
    return sig == signature


class Nexus(object):
    """
    Writer class to handle Nexus/HDF5 data

    Manages:

    - entry

        - pyFAI-subentry

            - detector

    TODO: make it thread-safe !!!
    """

    def __init__(self, filename, mode=None, creator=None):
        """
        Constructor

        :param filename: name of the hdf5 file containing the nexus
        :param mode: can be 'r', 'a', 'w', '+' ....
        :param creator: set as attr of the NXroot
        """
        self.filename = os.path.abspath(filename)
        self.mode = mode
        if not h5py:
            logger.error("h5py module missing: NeXus not supported")
            raise RuntimeError("H5py module is missing")

        pre_existing = os.path.exists(self.filename)
        if self.mode is None:
            if pre_existing:
                self.mode = "r"
            else:
                self.mode = "a"

        if self.mode == "r" and h5py.version.version_tuple >= (2, 9):
            self.file_handle = open(self.filename, mode=self.mode + "b")
            self.h5 = h5py.File(self.file_handle, mode=self.mode)
        else:
            self.file_handle = None
            self.h5 = h5py.File(self.filename, mode=self.mode)
        self.to_close = []

        if not pre_existing or "w" in mode:
            self.h5.attrs["NX_class"] = "NXroot"
            self.h5.attrs["file_time"] = get_isotime()
            self.h5.attrs["file_name"] = self.filename
            self.h5.attrs["HDF5_Version"] = h5py.version.hdf5_version
            self.h5.attrs["creator"] = creator or self.__class__.__name__

    def close(self, end_time=None):
        """
        close the filename and update all entries
        """
        if self.mode != "r":
            end_time = get_isotime(end_time)
            for entry in self.to_close:
                entry["end_time"] = end_time
            self.h5.attrs["file_update_time"] = get_isotime()
        self.h5.close()
        if self.file_handle:
            self.file_handle.close()

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def flush(self):
        if self.h5:
            self.h5.flush()

    def get_entry(self, name):
        """
        Retrieves an entry from its name

        :param name: name of the entry to retrieve
        :return: HDF5 group of NXclass == NXentry
        """
        for grp_name in self.h5:
            if grp_name == name:
                grp = self.h5[grp_name]
                if isinstance(grp, h5py.Group) and \
                   ("start_time" in grp) and  \
                   self.get_attr(grp, "NX_class") == "NXentry":
                        return grp

    def get_entries(self):
        """
        retrieves all entry sorted the latest first.

        :return: list of HDF5 groups
        """
        entries = [(grp, from_isotime(self.h5[grp + "/start_time"][()]))
                   for grp in self.h5
                   if isinstance(self.h5[grp], h5py.Group) and
                   ("start_time" in self.h5[grp]) and
                   self.get_attr(self.h5[grp], "NX_class") == "NXentry"]
        entries.sort(key=lambda a: a[1], reverse=True)  # sort entries in decreasing time
        return [self.h5[i[0]] for i in entries]

    def find_detector(self, all=False):
        """
        Tries to find a detector within a NeXus file, takes the first compatible detector

        :param all: return all detectors found as a list
        """
        result = []
        for entry in self.get_entries():
            for instrument in self.get_class(entry, "NXsubentry") + self.get_class(entry, "NXinstrument"):
                for detector in self.get_class(instrument, "NXdetector"):
                    if all:
                        result.append(detector)
                    else:
                        return detector
        return result

    def new_entry(self, entry="entry", program_name="pyFAI",
                  title=None, force_time=None, force_name=False):
        """
        Create a new entry

        :param entry: name of the entry
        :param program_name: value of the field as string
        :param title: description of experiment as str
        :param force_time: enforce the start_time (as string!)
        :param force_name: force the entry name as such, without numerical suffix.
        :return: the corresponding HDF5 group
        """
        if not force_name:
            nb_entries = len(self.get_entries())
            entry = "%s_%04i" % (entry, nb_entries)
        entry_grp = self.h5
        for i in entry.split("/"):
            if i:
                entry_grp = entry_grp.require_group(i)
        self.h5.attrs["default"] = entry
        entry_grp.attrs["NX_class"] = "NXentry"
        if title is not None:
            entry_grp["title"] = str(title)
        entry_grp["program_name"] = str(program_name)
        if force_time:
            entry_grp["start_time"] = str(force_time)
        else:
            entry_grp["start_time"] = get_isotime()
        self.to_close.append(entry_grp)
        return entry_grp

    def new_instrument(self, entry="entry", instrument_name="id00",):
        """
        Create an instrument in an entry or create both the entry and the instrument if
        """
        if not isinstance(entry, h5py.Group):
            entry = self.new_entry(entry)
        return self.new_class(entry, instrument_name, "NXinstrument")
#        howto external link
        # myfile['ext link'] = h5py.ExternalLink("otherfile.hdf5", "/path/to/resource")

    def new_class(self, grp, name, class_type="NXcollection"):
        """
        create a new sub-group with  type class_type
        :param grp: parent group
        :param name: name of the sub-group
        :param class_type: NeXus class name
        :return: subgroup created
        """
        sub = grp.require_group(name)
        sub.attrs["NX_class"] = str(class_type)
        return sub

    def new_detector(self, name="detector", entry="entry", subentry="pyFAI"):
        """
        Create a new entry/pyFAI/Detector

        :param detector: name of the detector
        :param entry: name of the entry
        :param subentry: all pyFAI description of detectors should be in a pyFAI sub-entry
        """
        entry_grp = self.new_entry(entry)
        pyFAI_grp = self.new_class(entry_grp, subentry, "NXsubentry")
        pyFAI_grp["definition_local"] = str("pyFAI")
        pyFAI_grp["definition_local"].attrs["version"] = str(version)
        det_grp = self.new_class(pyFAI_grp, name, "NXdetector")
        return det_grp

    def get_class(self, grp, class_type="NXcollection"):
        """
        return all sub-groups of the given type within a group

        :param grp: HDF5 group
        :param class_type: name of the NeXus class
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Group) and
                self.get_attr(grp[name], "NX_class") == class_type]
        return coll

    def get_dataset(self, grp, attr=None, value=None):
        """return list of dataset of the group matching 
        the given attribute having the given value 

        :param grp: HDF5 group
        :param attr: name of an attribute
        :param value: requested value for the attribute
        :return: list of dataset
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Dataset) and
                self.get_attr(grp[name], attr) == value]
        return coll

    def get_default_NXdata(self):
        """Return the default plot configured in the nexus structure.
        
        :return: the group with the default plot or None if not found
        """
        entry_name = self.h5.attrs.get("default")
        if entry_name:
            entry_grp = self.h5.get(entry_name)
            nxdata_name = entry_grp.attrs.get("default")
            if nxdata_name:
                if nxdata_name.startswith("/"):
                    return self.h5.get(nxdata_name)
                else:
                    return entry_grp.get(nxdata_name)

    def deep_copy(self, name, obj, where="/", toplevel=None, excluded=None, overwrite=False):
        """
        perform a deep copy:
        create a "name" entry in self containing a copy of the object

        :param where: path to the toplevel object (i.e. root)
        :param  toplevel: firectly the top level Group
        :param excluded: list of keys to be excluded
        :param overwrite: replace content if already existing
        """
        if (excluded is not None) and (name in excluded):
            return
        if not toplevel:
            toplevel = self.h5[where]
        if isinstance(obj, h5py.Group):
            if name not in toplevel:
                grp = toplevel.require_group(name)
                for k, v in obj.attrs.items():
                        grp.attrs[k] = v
        elif isinstance(obj, h5py.Dataset):
            if name in toplevel:
                if overwrite:
                    del toplevel[name]
                    logger.warning("Overwriting %s in %s", toplevel[name].name, self.filename)
                else:
                    logger.warning("Not overwriting %s in %s", toplevel[name].name, self.filename)
                    return
            toplevel[name] = obj[()]
            for k, v in obj.attrs.items():
                toplevel[name].attrs[k] = v

    @classmethod
    def get_attr(cls, dset, name, default=None):
        """Return the attribute of the dataset

        Handles the ascii -> unicode issue in python3 #275

        :param dset: a HDF5 dataset (or a group)
        :param name: name of the attribute
        :param default: default value to be returned
        :return: attribute value decoded in python3 or default
        """
        dec = default
        if name in dset.attrs:
            raw = dset.attrs[name]
            if (sys.version_info[0] > 2) and ("decode" in dir(raw)):
                dec = raw.decode()
            else:
                dec = raw
        return dec


def save_rebin(filename, beamline="id22", name="id22rebin", topas=None, res=None, start_time=None):
    """Save rebinned data with external links to input data
    
    :param filename:
    :param beamline:
    :param name: program name
    :param topas: dict with topas configuration
    :param res: 3/4-tuple with results
    """
    weights = None
    with  Nexus(filename, mode="w", creator=name) as nxs:
        entry = nxs.new_entry(entry="entry", program_name=name,
                              title=None, force_time=start_time, force_name=False)
        process_grp = nxs.new_class(entry, "id22rebin", class_type="NXprocess")
        process_grp["program"] = name
        process_grp["sequence_index"] = 1
        process_grp["version"] = version
        process_grp["date"] = get_isotime()
        process_grp.create_dataset("argv", data=numpy.array(sys.argv, h5py.string_dtype("utf8"))).attrs["help"] = "Command line arguments"
        process_grp.create_dataset("cwd", data=os.getcwd()).attrs["help"] = "Working directory"

        if topas:
            topas_grp = nxs.new_class(process_grp, "topas")
            for k, v in topas.items():
                topas_grp.create_dataset(k, data=v).attrs["unit"] = "rad"

        if res:
            data_grp = nxs.new_class(process_grp, "data", "NXdata")
            tth_ds = data_grp.create_dataset("2th", data=res[0], **CMP)
            tth_ds.attrs["unit"] = "deg"
            
            sum_ds = data_grp.create_dataset("I_sum", data=res[1], **CMP)
            sum_ds.attrs["interpretation"] = "spectrum"
            norm_ds = data_grp.create_dataset("norm", data=res[2], **CMP)
            norm_ds.attrs["interpretation"] = "spectrum"
            
            scale = numpy.atleast_2d(numpy.median(res[2], axis=-1)).T
            with numpy.errstate(divide='ignore', invalid='ignore'):
                I = scale * res[1] / res[2]
            Ima_ds = data_grp.create_dataset("I_MA", data=I , **CMP)
            Ima_ds.attrs["interpretation"] = "spectrum"
            
            if topas:
                Ima_ds.attrs["axes"] = ["offset", "2th"]
                offset_ds = data_grp.create_dataset("offset", data=numpy.rad2deg(topas["offset"]))
                offset_ds.attrs["unit"] = "deg"
                weights = numpy.array(topas.get("scale"))
            else:
                Ima_ds.attrs["axes"] = [".", "2th"]
                
            if weights is None:
                weights = numpy.ones(res[1].shape[0])
            weights /= weights.sum() # normalize wights
            weights = numpy.atleast_2d(weights).T
            # print(weights.shape)
            # print(scale.shape)
            I_avg =  (scale * weights * res[1]).sum(axis=0) / (weights*res[2]).sum(axis=0)
            I_ds = data_grp.create_dataset("I_avg", data=I_avg , **CMP)
            I_ds.attrs["interpretation"] = "spectrum"
            I_ds.attrs["axes"] = ["2th"]
            #TODO: perform uncertainty proagation using https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
            
            data_grp.attrs["signal"] = posixpath.basename(I_ds.name)
            entry.attrs["default"] = data_grp.name
            if len(res) >= 4:
                debug_ds = data_grp.create_dataset("cycles", data=res[3] , **CMP)
                debug_ds.attrs["interpretation"] = "image"
                debug_ds.attrs["info"] = "Number of refinement cycle to converge 2theta"

