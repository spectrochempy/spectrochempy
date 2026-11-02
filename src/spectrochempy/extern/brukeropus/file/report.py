# ruff: noqa
import re

from spectrochempy.extern.brukeropus.file.block import FileBlock

__docformat__ = "google"


class Report:
    """Class containing data from an OPUS file report block.

    OPUS files may contain data in the form of a report block.  The report block format is generic, and can be used to
    store a variety of data types. The entire contents of a report can be viewed by printing the class to the console:
    print(report). All of the data in the report is accessible through class attributes and indexing.

    Args:
        block: parsed `FileBlock` instance of a report block

    Attributes:
        title: (`str`) title of report
        properties: (`dict`) top-level properties of the report in the form of key, val pairs
        table: (`ReportTable`) tabular data with header and title stored in the top-level of the report
        sub: (`list`(`ReportTable`)) list of subreports (tabular data) stored in the report

    Indexing:
        The class supports indexing as a shortcut to accessing certain data within the report.
        int: subreports can be accessed by indexing an instance of a report with an integer number.
            report[0] == report.sub[0] (i.e. access first subreport)
        str: The values of report properties can be accessed by indexing an instance of a report with a string (case
            insensitive). report['version'] == report.properties['Version'] (i.e. access 'version' property of report)
    """

    def __getitem__(self, item):
        if type(item) is int and hasattr(self, "sub"):
            return self.sub[item]
        if type(item) is str and item.lower() in [
            k.lower() for k in self.properties.keys()
        ]:
            key = [k for k in self.properties.keys() if k.lower() == item.lower()][0]
            return self.properties[key]
        raise KeyError(
            str(item)
            + """ not a valid key. Key can be integer (to reference subreport) or a key from
                           .properties.keys()"""
        )

    def __init__(self, block: FileBlock):
        self.block = block
        data = self.block.data
        self.title = data["header"]["tit"]
        self.table = ReportTable(
            info=data["info"], data=data["data"], title=data["header"]["e00"]
        )
        if "h00" in data["info"].keys():
            self.properties = dict()
            for key in data["info"].keys():
                if re.search("h[0-9][0-9]", key):
                    try:
                        val = data["info"]["v" + key[1:]]
                    except:
                        val = None
                    self.properties[data["info"][key]] = val
        self.sub = []
        if "subreports" in data.keys():
            for i, subreport in enumerate(data["subreports"]):
                title = data["info"]["u" + f"{i:02}"]
                self.sub.append(ReportTable(title=title, **subreport))
        self._data = data
        self.block.data = None

    def __repr__(self):
        contents = []
        if hasattr(self, "properties"):
            contents.append(f"properties ({len(self.properties)})")
        if hasattr(self, "table"):
            contents.append("table (" + f"{self.table.num_cols}x{self.table.num_rows})")
        if hasattr(self, "sub"):
            contents.append(f"sub ({len(self.sub)})")
        return f"Report({self.title}: " + ", ".join(contents) + ")"

    def __str__(self, width=100, max_col_width=25, float_digits=3):
        lines = ["=" * width, f"{self.title:^{width}}", "_" * width]
        if hasattr(self, "properties"):
            key_width = max(len(key) for key in self.properties.keys())
            for k, v in self.properties.items():
                lines.append(f"{k:>{key_width}}" + ": " + str(v))
            lines.append("\n")
        table_str = self.table.__str__(
            width=width, max_width=max_col_width, float_digits=float_digits
        )
        lines = lines + table_str.splitlines()
        if hasattr(self, "sub"):
            for i, s in enumerate(self.sub):
                lines.append("Subreport " + str(i))
                s_str = s.__str__(
                    width=width, max_width=max_col_width, float_digits=float_digits
                )
                lines = lines + s_str.splitlines()
        lines.append("=" * width)
        return "\n".join(lines)


class ReportTable:
    """Class containing table data from an OPUS file report block.

    OPUS file report blocks generally contain one or more tables of data. This class is used to store and provide an
    interface for accessing that tabular data.

    Attributes:
        title: (str) title of the table
        num_cols: (int) number of columns in the table
        num_rows: (int) number of rows in the table
        header: (list(str)) list of header labels for each row of data
        values: (list(list(vals))) 2D list of tabular data. Can be indexed: values[row][col]

    Indexing:
        The class supports indexing as a shortcut to accessing certain data within the report.
        int: indexing the table with an integer returns the corresponding row of values in the table.
            table[0] == table.values[0] (i.e. first row of data in the table)
        str: indexing the table with a string returns the row of data with matching header label (case insensitive).
            table['type'] == table.values[table.header.index('Type')]
    """

    def __getitem__(self, item):
        if type(item) is int:
            return self.values[item]
        if type(item) is str and item.lower() in [h.lower() for h in self.header]:
            idx = [i for i, h in enumerate(self.header) if h.lower() == item.lower()][0]
            return self.values[idx]
        raise KeyError(
            str(item) + " not a valid key. For a list of valid keys, use: .headers"
        )

    def __init__(self, info: dict, data: list, title: str):
        self._info = info
        self._data = data
        self.title = title
        self.num_cols = len(data)
        self.num_rows = len(data[0])
        self.header = [info["s" + f"{row:02}"] for row in range(self.num_rows)]
        self.values = [list(row) for row in zip(*data)]

    def __repr__(self):
        return (
            self.title
            + " (table): cols: "
            + str(self.num_cols)
            + ", rows: "
            + str(self.num_rows)
        )

    def _str(self, val, max_width=25, float_digits=3):
        if type(val) is float:
            s = f"{val:.6g}"
        else:
            s = str(val)
        if len(s) > max_width:
            s = s[: max_width - 3] + "..."
        return s

    def __str__(self, width=100, max_width=25, float_digits=3):
        label = self.__repr__()[:width]
        lines = [label, "-" * width]
        h_width = max(len(h) for h in self.header)
        vals_width = max([width - h_width - 2, 20])
        if self.num_cols == 1:
            col_widths = [vals_width]
        else:
            col_widths = [
                max([len(self._str(row[col])) for row in self.values])
                for col in range(self.num_cols)
            ]
            if sum(col_widths) > vals_width:
                col_widths = [
                    int(
                        min(
                            [(vals_width - 2 * (self.num_cols - 1)) // self.num_cols, c]
                        )
                    )
                    for c in col_widths
                ]
        for i, h in enumerate(self.header):
            lines.append(
                f"{h:>{h_width}}"
                + ": "
                + "  ".join(
                    [
                        f"{self._str(v, max_width=col_widths[j], float_digits=float_digits):<{col_widths[j]}}"
                        for j, v in enumerate(self.values[i])
                    ]
                )
            )
        lines.append("-" * width)
        return "\n".join(lines)
