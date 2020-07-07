import copy
import json
import logging
import pprint
import re
from collections import defaultdict
from enum import Enum

from pydicom import Dataset
from pydicom.datadict import dictionary_has_tag, dictionary_keyword
from pydicom.tag import Tag

logger = logging.getLogger(__name__)


class AccompanyingDicomData:

    class DataType(Enum):
        CT = 1
        Dose = 2
        Struct = 3
        LET = 4
        common_CT = 5
        common_all = 6

    def __init__(self, ct_datasets=[], dose_dataset=None, structure_dataset=None):
        logger.info("Creating Accompanying DICOM data")
        logger.debug("Accessing {:d} CT datasets".format(len(ct_datasets)))
        if dose_dataset:
            logger.debug("Accessing dose datasets")
        if structure_dataset:
            logger.debug("Accessing structure datasets")

        # save tag values
        self.headers_datasets = {}
        self.data_datasets = {}
        if dose_dataset:
            self.headers_datasets[self.DataType.Dose] = Dataset(copy.deepcopy(dose_dataset.file_meta))
            self.data_datasets[self.DataType.Dose] = Dataset(copy.deepcopy(dose_dataset))

            # remove Pixel Data to save space
            del self.data_datasets[self.DataType.Dose].PixelData

        if ct_datasets:
            self.headers_datasets[self.DataType.CT] = \
                dict((dataset.InstanceNumber, Dataset(copy.deepcopy(dataset.file_meta))) for dataset in ct_datasets)
            self.data_datasets[self.DataType.CT] = {}
            for dataset in ct_datasets:
                self.data_datasets[self.DataType.CT][dataset.InstanceNumber] = Dataset(copy.deepcopy(dataset))

                # remove Pixel Data to save space
                del self.data_datasets[self.DataType.CT][dataset.InstanceNumber].PixelData

        if structure_dataset:
            self.headers_datasets[self.DataType.Struct] = Dataset(copy.deepcopy(structure_dataset.file_meta))
            self.data_datasets[self.DataType.Struct] = Dataset(copy.deepcopy(structure_dataset))

            # remove Contour Data to save space
            for i, contour in enumerate(self.data_datasets[self.DataType.Struct].ROIContourSequence):
                for j, sequence in enumerate(contour.ContourSequence):
                    del self.data_datasets[self.DataType.Struct].ROIContourSequence[i].ContourSequence[j].ContourData

        self.update_common_tags_and_values()

    def update_common_tags_and_values(self):
        """
        TODO
        :return:
        """

        all_header_datasets = list(self.headers_datasets.get(self.DataType.CT, {}).values())
        if self.DataType.Struct in self.headers_datasets.keys():
            all_header_datasets.append(self.headers_datasets[self.DataType.Struct])
        if self.DataType.Dose in self.headers_datasets.keys():
            all_header_datasets.append(self.headers_datasets[self.DataType.Dose])

        all_data_datasets = list(self.data_datasets.get(self.DataType.CT, {}).values())
        if self.DataType.Struct in self.data_datasets.keys():
            all_data_datasets.append(self.data_datasets[self.DataType.Struct])
        if self.DataType.Dose in self.data_datasets.keys():
            all_data_datasets.append(self.data_datasets[self.DataType.Dose])

        # list of common tags+values for all datasets (header and data file)
        self.all_datasets_header_common = self.find_common_tags_and_values(all_header_datasets)
        self.all_datasets_data_common = self.find_common_tags_and_values(all_data_datasets)
        self.all_datasets_data_common.discard(Tag('PixelData'))  # TODO check if it is working

        # list of common tags+values for CT datasets (header and data file)
        self.ct_datasets_header_common = self.find_common_tags_and_values(
            list(self.headers_datasets.get(self.DataType.CT, {}).values()))
        self.ct_datasets_data_common = self.find_common_tags_and_values(
            list(self.data_datasets.get(self.DataType.CT, {}).values())
        )
        self.ct_datasets_data_common.discard(Tag('PixelData'))

        # list of file specific tags (without values!) for CT datasets (header and data file)
        self.ct_datasets_header_specific = self.find_tags_with_specific_values(
            list(self.headers_datasets.get(self.DataType.CT, {}).values()))
        self.ct_datasets_data_specific = self.find_tags_with_specific_values(
            list(self.data_datasets.get(self.DataType.CT, {}).values()))
        self.ct_datasets_data_specific.discard(Tag('PixelData'))

    @staticmethod
    def find_common_tags(list_of_datasets=[], access_method=lambda x: x):
        """
        TODO
        :param list_of_datasets:
        :param access_method:
        :return: set of common tags
        """
        common_tags = set()
        if list_of_datasets:
            common_tags = set(access_method(list_of_datasets[0]).keys())
            for dataset in list_of_datasets[1:]:
                common_tags.intersection_update(access_method(dataset).keys())
        return common_tags

    @classmethod
    def find_common_tags_and_values(cls, list_of_datasets=[], access_method=lambda x: x):
        """
        TODO
        :param list_of_datasets:
        :param access_method:
        :return: set of tuples (tag and values) with common values
        """
        common_tags_and_values = set()
        if list_of_datasets:
            common_tags = cls.find_common_tags(list_of_datasets, access_method)
            first_dataset = access_method(list_of_datasets[0])
            for tag in common_tags:
                if all([first_dataset[tag] == access_method(dataset)[tag] for dataset in list_of_datasets[1:]]):
                    common_tags_and_values.add((tag, first_dataset[tag].repval))
        return common_tags_and_values

    @classmethod
    def find_tags_with_specific_values(cls, list_of_datasets=[], access_method=lambda x: x):
        """
        TODO
        :param list_of_datasets:
        :param access_method:
        :return: set of common tags
        """
        tags_with_specific_values = set()
        if list_of_datasets:
            common_tags = cls.find_common_tags(list_of_datasets, access_method)
            first_dataset = access_method(list_of_datasets[0])
            for tag in common_tags:
                if not all([first_dataset[tag] == access_method(dataset)[tag] for dataset in list_of_datasets[1:]]):
                    tags_with_specific_values.add(tag)
        return tags_with_specific_values

    def to_comment(self):
        """

        :return:
        """

        self.update_common_tags_and_values()

        # generate data to save
        ct_json_dict = {}
        if self.DataType.CT in self.headers_datasets:
            ct_json_dict['header'] = {}
            first_instance_id = list(self.headers_datasets[self.DataType.CT].keys())[0]
            ct_json_dict['header']['common'] = Dataset(dict(
                (tag_name, self.headers_datasets[self.DataType.CT][first_instance_id][tag_name])
                for tag_name, _ in self.ct_datasets_header_common
            )).to_json_dict()

            ct_json_dict['header']['specific'] = {}
            for instance_id, dataset in self.headers_datasets[self.DataType.CT].items():
                ct_json_dict['header']['specific'][instance_id] = \
                    Dataset(dict(
                        (tag_name, dataset[tag_name]) for tag_name in self.ct_datasets_header_specific
                    )).to_json_dict()

        if self.DataType.CT in self.data_datasets:
            ct_json_dict['data'] = {}
            first_instance_id = list(self.data_datasets[self.DataType.CT].keys())[0]

            ct_json_dict['data']['common'] = Dataset(dict(
                (tag_name, self.data_datasets[self.DataType.CT][first_instance_id][tag_name])
                for tag_name, _ in self.ct_datasets_data_common
            )).to_json_dict()

            ct_json_dict['data']['specific'] = {}
            for instance_id, dataset in self.data_datasets[self.DataType.CT].items():
                ct_json_dict['data']['specific'][instance_id] = \
                    Dataset(dict(
                        (tag_name, dataset[tag_name]) for tag_name in self.ct_datasets_data_specific
                    )).to_json_dict()

        dose_json_dict = {}
        if self.DataType.Dose in self.headers_datasets:
            dose_json_dict['header'] = self.headers_datasets[self.DataType.Dose].to_json_dict()
        if self.DataType.Dose in self.data_datasets:
            dose_json_dict['data'] = self.data_datasets[self.DataType.Dose].to_json_dict()

        struct_json_dict = {}
        if self.DataType.Struct in self.headers_datasets:
            struct_json_dict['header'] = self.headers_datasets[self.DataType.Struct].to_json_dict()
        if self.DataType.Struct in self.data_datasets:
            struct_json_dict['data'] = self.data_datasets[self.DataType.Struct].to_json_dict()

        # save the result string
        result = ""
        if ct_json_dict or struct_json_dict or dose_json_dict:
            result += "#############################################################\n"
            result += "#############################################################\n"
            result += "####### This file was created from a DICOM data #############\n"
            result += "#############################################################\n"
            result += "#############################################################\n"

        if ct_json_dict:
            result += "####### CT begins #############\n"
            pretty_string = pprint.pformat(ct_json_dict, width=180)
            no_of_lines = len(pretty_string.splitlines())
            for line_no, line in enumerate(pretty_string.splitlines()):
                result += "#@CT@ line {:d} / {:d} : {:s}\n".format(line_no, no_of_lines, line)
            result += "####### CT ends #############\n"

        if struct_json_dict:
            result += "####### Struct begins #############\n"
            pretty_string = pprint.pformat(struct_json_dict, width=180)
            no_of_lines = len(pretty_string.splitlines())
            for line_no, line in enumerate(pretty_string.splitlines()):
                result += "#@Struct@ line {:d} / {:d} : {:s}\n".format(line_no, no_of_lines, line)
            result += "####### Struct ends #############\n"

        if dose_json_dict:
            result += "####### Dose begins #############\n"
            pretty_string = pprint.pformat(dose_json_dict, width=180)
            no_of_lines = len(pretty_string.splitlines())
            for line_no, line in enumerate(pretty_string.splitlines()):
                result += "#@Dose@ line {:d} / {:d} : {:s}\n".format(line_no, no_of_lines, line)
            result += "####### Dose ends #############\n"

        return result

#    @jit
    def from_comment(self, parsed_str):
        logger.debug("Starting to parse DICOM comment data")

        re_exp = "#@(?P<type>.+)@ line (?P<line_no>.+) \/ (?P<line_total>.+) : (?P<content>.+)"  # NOQA: W605
        regex = re.compile(re_exp)

        content_by_type = defaultdict(list)
        for i, line in enumerate(parsed_str):
            match = regex.search(line)
            if match:
                content_by_type[match.group('type')].append(match.group('content').replace('\'', '\"'))

        logger.debug("Stopped to parse DICOM comment data")

        if 'CT' in content_by_type:
            logger.debug("Restructuring CT data, JSON loading")
            ct_dicts = json.loads("\n".join(content_by_type['CT']))
            logger.debug("Restructuring CT data, JSON loaded")
            if ct_dicts['data']['specific']:
                self.data_datasets[self.DataType.CT] = {}
            for instance_id, dataset_dict in ct_dicts['data']['specific'].items():
                self.data_datasets[self.DataType.CT][instance_id] = Dataset().from_json(dataset_dict)
                self.data_datasets[self.DataType.CT][instance_id].update(
                    Dataset().from_json(ct_dicts['data']['common'])
                )

            logger.debug("data dictionaries loaded {}".format(len(ct_dicts['data']['specific'])))

            if ct_dicts['header']['specific']:
                self.headers_datasets[self.DataType.CT] = {}
            for instance_id, dataset_dict in ct_dicts['header']['specific'].items():
                # instance_id : str
                self.headers_datasets[self.DataType.CT][instance_id] = Dataset().from_json(dataset_dict)
                self.headers_datasets[self.DataType.CT][instance_id].update(
                    Dataset().from_json(ct_dicts['header']['common'])
                )
            logger.debug("header dictionaries loaded")

        if 'Struct' in content_by_type:
            logger.debug("Restructuring Struct data, JSON loading")
            struct_dicts = json.loads("\n".join(content_by_type['Struct']))
            self.headers_datasets[self.DataType.Struct] = Dataset().from_json(struct_dicts['header'])
            self.data_datasets[self.DataType.Struct] = Dataset().from_json(struct_dicts['data'])

        if 'Dose' in content_by_type:
            logger.debug("Restructuring Dose data, JSON loading")
            dose_dicts = json.loads("\n".join(content_by_type['Dose']))
            self.headers_datasets[self.DataType.Dose] = Dataset().from_json(dose_dicts['header'])
            self.data_datasets[self.DataType.Dose] = Dataset().from_json(dose_dicts['data'])

        self.update_common_tags_and_values()

    def __str__(self):

        def nice_tag_name(tag):
            if dictionary_has_tag(tag):
                return dictionary_keyword(tag_name)
            else:
                return ""

        result = "all datasets\n"
        result += "\theader (file meta) common tags and values:\n"
        for (tag_name, tag_value) in sorted(self.all_datasets_header_common, key=lambda x: x[0]):
            result += "\t\t{:s} {:s} = {:s}\n".format(str(tag_name), nice_tag_name(tag_name), str(tag_value))
        result += "\tdata common tags and values:\n"
        for (tag_name, tag_value) in sorted(self.all_datasets_data_common, key=lambda x: x[0]):
            result += "\t\t{:s} {:s} = {:s}\n".format(str(tag_name), nice_tag_name(tag_name), str(tag_value))

        result += "CT datasets\n"
        result += "\theader (file meta) common tags and values:\n"
        for (tag_name, tag_value) in sorted(self.ct_datasets_header_common, key=lambda x: x[0]):
            result += "\t\t{:s} {:s} = {:s}\n".format(str(tag_name), nice_tag_name(tag_name), str(tag_value))
        result += "\tdata common tags and values:\n"
        for (tag_name, tag_value) in sorted(self.ct_datasets_data_common, key=lambda x: x[0]):
            result += "\t\t{:s} {:s} = {:s}\n".format(str(tag_name), nice_tag_name(tag_name), str(tag_value))

        result += "CT datasets\n"
        result += "\theader (file meta) tags with specific values:\n"
        for tag_name in sorted(self.ct_datasets_header_specific):
            result += "\t\t{:s} {:s}\n".format(str(tag_name), nice_tag_name(tag_name))
        result += "\tdata tags with specific values:\n"
        for tag_name in sorted(self.ct_datasets_data_specific):
            result += "\t\t{:s} {:s}\n".format(str(tag_name), nice_tag_name(tag_name))

        return result
