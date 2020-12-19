# feed the preference for plotting from rcParams
import pathlib
from textwrap import indent
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cycler

from traitlets import Bool, Unicode, List, Tuple, Integer, Int, Float, Enum, Any, observe, All

template = r'''## ############################################ ##
## DO NOT MODIFY                                ##
## Module generated using .ci/scripts/makeprefs ##
##################################################

import matplotlib as mpl
from matplotlib import cycler
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from traitlets import Bool, Unicode, Tuple, Integer, Float, Enum

from spectrochempy.utils import MetaConfigurable

class MatplotlibPreferences(MetaConfigurable):
    """
    This is a port of matplotlib.rcParams to our configuration system (traitlets)
    
    """

    name = Unicode("MatplotlibPreferences")
    description = Unicode("Options for Matplotlib")
    updated = Bool(False)

    # ------------------------------------------------------------------------------------------------------------------  
    # Configuration entries  
    # ------------------------------------------------------------------------------------------------------------------
    
{list_of_traits}

    # ..................................................................................................................
    def __init__(self, **kwargs):
    
        super().__init__(jsonfile='MatplotlibPreferences', **kwargs)
        
    # Alias
    # -----
    @property
    def colormap(self):
        return self.image_cmap
        
    @colormap.setter
    def colormap(self, val):
        self.image_cmap = val
    
'''

def get_fontsize(fontsize):
    if fontsize == 'None':
        return float(mpl.rcParams['font.size'])
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, 'Text')
    try:
        t.set_fontsize(fontsize)
        return str(round(t.get_fontsize(), 2))
    except:
        return fontsize

def get_color(color):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    c = [f'C{i}' for i in range(10)]
    if color in c:
        return f"'{colors[c.index(color)]}'"
    else:
        return f"'{color}'"

plt.style.use('scpy')

lst = []
comments=[]

with open(mpl.matplotlib_fname(), 'r') as f:
    content = f.readlines()
for line in content[:]:
    if not line.strip() or line == '#\n' or line.startswith('##') or line.startswith('# '):
        content.remove(line)
for i, rawline in enumerate(content):
    line = rawline
    if line.startswith('#'):
        line = line[1:]
    line = line.strip()
    if line.startswith('#'):
        lst[-1][2] = lst[-1][2] + '\n' + line[1:]
        comments[-1].append(rawline[1:])
    else:
        item = [""] * 3
        if ':' in line:
            item[:1] = line.split(':')
        elif '=' in line:
            item[:1] = line.split('=')
        else:
            continue
        if '#' in item[1]:
            item[1:] = item[1].split('#')
        for i, part in enumerate(item):
            item[i] = item[i].strip()
        lst.append(item)
        comments.append([rawline[1:]])


# try to convert these items in spectrochempy pref

text = ""
for id, item in enumerate(lst):

    name = value = obj = None

    name = item[0]

    try:
        value = mpl.rcParams[name] #item[1]
    except KeyError:
        # print( "no key ", name, ' in rcParams')
        continue
    help = item[2]

    # try to determine the type and kind
    val = ''
    kind = ""

    # case of sizes:
    if "size" in name and "figsize" not in name and 'papersize' not in name:
        try:
            val = get_fontsize(value)
            obj = 'Unicode'
        except:
            pass

    elif name.endswith('marker'):
        obj = 'Enum'
        val = f"list(Line2D.markers.keys()), default_value='{value}'"
    elif name.endswith('linestyle'):
        obj = 'Enum'
        val = f"list(Line2D.lineStyles.keys()), default_value='{value}'"
    elif name.endswith('color') and not 'force_' in name:
        obj = 'Unicode'
        val = get_color(value)
        kind = "color"
    elif name.endswith('cmap'):
        obj = 'Enum'
        val = f"plt.colormaps(), default_value='{value}'"
    else:
        try:
            if isinstance(value, str):
                val = eval(value)
            else:
                val = value
            if isinstance(val, bool):
                obj = 'Bool'
            elif isinstance(val, float):
                obj = 'Float'
            elif isinstance(val, int):
                obj = 'Integer'
            elif isinstance(val, (tuple, list)):
                val = tuple(value)
                obj = 'Tuple'
            elif isinstance(val, type(round)):
                val = f"'{val}'"
                obj = 'Unicode'
            elif val is None or isinstance(val, str):
                val = f"'{val}'".replace(r"\\", r"\\\\")
                obj = 'Unicode'

        except NameError:
            print(item, "NameError", value)
            value = value.replace(r"\\", r"\\\\").replace("'", '"')
            val = f"'{value}'"
            obj = 'Unicode'

        except SyntaxError:
            print(item, "SyntaxError", value)
            value = value.replace(r"\\", r"\\\\").replace("'", '"')
            val = f'"{value}"'
            obj = 'Unicode'

        except:
            val = value
            obj = 'Any'

        if obj is None:
            print()

    #val = f'mpl.rcParams["{name}"]'

    help = help.replace("'", '"').replace("    "," ").replace("   "," ").replace("  "," ")
    text += '\n'
    for  comment in comments[id]:
        text += f'## {comment}'
    if text[-1] != '\n':
        text += '\n'
    name_ = name.replace('_', '___').replace('.', '_').replace('-', '__')
    field = f"{name_} = {obj}({val}, help=r'''{help}''').tag(config=True, kind='{kind}')"
    try:
        # check result
        exec(field)
        eval(name_)
    except Exception as e:
        print(e)

    text += f'{field}\n'

text = template.format(list_of_traits = indent(text, '    '))

filename = pathlib.Path(__file__).parent.parent.parent / 'spectrochempy' / '~matplotlib_preferences.py'
filename.write_text(text)
