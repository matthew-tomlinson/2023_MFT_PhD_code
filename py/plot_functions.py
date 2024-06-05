

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings


from statsmodels.tsa.stattools import acf


# Local files
import misc_functions as misc_fns
import greek_roman as GR
#import plot_functions as plot_fns




#============
#============
#
# Functions
#
#============
#============


#### log_ret


def mkdir_export(path, mkdir=None):

    if mkdir is None:
        mkdir = True
    if mkdir:
        if os.path.exists(path)==False:
            os.mkdir(path)
    return path

def calc_root(root=None, mkdir=None):

    if mkdir is None:
        mkdir = True

    if root is None:
        root = f"./"

    return mkdir_export(path=root)


def calc_subs(subs=None, field=None, postsubs=None, app_field=None):

    if subs is None:
        subs = []
    if postsubs is None:
        postsubs = []

    if app_field is None:
        app_field = True
    if not app_field or field is None:
        field = []

    subs = misc_fns.make_iterable_array(subs)
    field = misc_fns.make_iterable_array(field)
    postsubs = misc_fns.make_iterable_array(postsubs)

    return np.concatenate([subs, field, postsubs])


def calc_subdir(subs=None, field=None, postsubs=None, app_field=None, root=None, mkdir=None):

    if mkdir is None:
        mkdir = True

    root = calc_root(root=root, mkdir=mkdir)
    subs = calc_subs(subs=subs, field=field, postsubs=postsubs, app_field=app_field)

    tdir = f"{root}"
    if len(subs)>0:
        subs = misc_fns.app_presuff(subs, suff='/', app_null=False)
    
        if mkdir:
            for S in subs:
                tdir = mkdir_export(path=f"{tdir}{S}", mkdir=mkdir)
        else:
            tdir = f"{tdir}{misc_fns.str_concatenate(subs)}"

    return tdir


def calc_dir(dir=None, subs=None, field=None, postsubs=None, app_field=None, root=None, mkdir=None):

    if mkdir is None:
        mkdir = True

    if dir is None:
        dir = calc_subdir(subs=subs, field=field, postsubs=postsubs, app_field=app_field, root=root, mkdir=mkdir)
    else:
        dir = mkdir_export(dir, mkdir=mkdir)

    return dir

def calc_write_address(dir=None, subs=None, root=None, file_nm=None, file_ext=None, mkdir=None):

    dir = calc_dir(dir=dir, subs=subs, root=root, mkdir=mkdir)

    if file_nm is None:
        file_nm = 'file'

    if file_ext is None:
        file_ext = ''

    return f"{dir}{file_nm}{file_ext}"


def fig_export(fig, height=None, width=None, dir=None, subs=None, root=None, mkdir=None, thesis=None, formats=None, dpi=None):

    if thesis is None:
        thesis = False

    if formats is None:
        formats = 'pdf'
    formats = misc_fns.make_iterable_array(formats)

    if dpi is not None:
        fig['fig'].set_dpi(dpi)

    if subs is None:
        subs = 'figures'

    dir = calc_dir(dir=dir, subs=subs, root=root, mkdir=mkdir)


    if width is None:
        if thesis:
            width = 5.7
        else:
            width = 7.0

    if height is None:
        height = 3.0


    fig['fig'].set_size_inches(width, height)
    fig['fig'].tight_layout()
    for fm in formats:
        fig['fig'].savefig(f"{dir}{fig['name']}.{fm}", format=fm)


    



def print_str_pw(print_str=None, do_print=None, do_write=None, write_mode=None, write_address=None, 
                 dir=None, subs=None, root=None, mkdir=None,
                 file_nm=None, file_ext=None):
    

    if print_str is None:
        print_str = []

    if do_print is None:
        do_print = True
    if do_write is None:
        do_write = False

    if do_print:
        for p in range(len(print_str)):
            print(print_str[p])

    if do_write:
        if write_mode is None:
            write_mode = "w"

        if file_ext is None:
            file_ext = ".txt"

        if write_address is None:
            write_address = calc_write_address(dir=dir, subs=subs, root=root, mkdir=mkdir,
                                        file_nm=file_nm, file_ext=file_ext)

        f = open(write_address, write_mode)
        for p in range(len(print_str)):
            f.write(f"{print_str[p]}\n")
        f.close()


def calc_ftab_str(tab_mfn_str, c_str, sizes=None, size=None, F_i=None, ftab_str=None, ftab_str_post=None):

    if sizes is None:
        sizes = ['footnote', 'script']
    sizes = misc_fns.make_iterable_array(sizes)
    if size is None:
        size = sizes[0]

    sizes = np.unique(np.concatenate([sizes, [size]]))

    if ftab_str is None:
        ftab_str = []
    
    if ftab_str_post is None:
        ftab_str_post = []

    if F_i is None:
        TAB_i = tab_mfn_str.find('TAB')
        bs_i = tab_mfn_str.find('\\')

        if TAB_i>=0:
            F_i = TAB_i
        elif bs_i>=0:
            F_i = bs_i
        else:
            F_i = 0

    ftab_mfn_str = f"{tab_mfn_str[:F_i]}F{tab_mfn_str[F_i:]}"
    
    ftab_str.append(f"\\newcommand{{{ftab_mfn_str}}}{{")
    for s in sizes:
        if s==size:
            void_str = ""
        else:
            void_str = "%"
        ftab_str.append(f"{void_str}\\{s}size")
    #ftab_str.append(f"\\begin{{tabular*}}{{\\textwidth}}{{{c_str}}}")
    ftab_str.append(f"\\begin{{tabular}}{{{c_str}}}")
    ftab_str.append(f"\\hline\n\\hline")
    ftab_str.append(tab_mfn_str)
    ftab_str.append(f"\\hline\n\\hline")
    #ftab_str.append(f"\\end{{tabular*}}")
    ftab_str.append(f"\\end{{tabular}}")
    if len(ftab_str_post)>0:
        ftab_str.append(misc_fns.str_concatenate(ftab_str_post, fill_str='\n', fill_str_void=-1))
    ftab_str.append(f"}}")
    ftab_str.append(f"")

    return ftab_str



def calc_tex_vect_fn(pltf=None, use_tex=None, tex_vect=None):


    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return misc_fns.calc_tex_vect_fn(pltf=pltf, use_tex=use_tex, tex_vect=tex_vect)


def calc_tex_vect_ps(tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return calc_tex_vect_ps(tex_vect=None, app_tex_vect=None, pltf=pltf, use_tex=use_tex)


def calc_tex_num_fn(pltf=None, use_tex=None, tex_num=None):


    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return misc_fns.calc_tex_num_fn(pltf=pltf, use_tex=use_tex, tex_num=tex_num)


def calc_tex_num_ps(tex_num=None, app_tex_num=None, pltf=None, use_tex=None):

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return misc_fns.calc_tex_num_ps(tex_num=tex_num, app_tex_num=app_tex_num, pltf=pltf, use_tex=use_tex)


def calc_tex_hphantom_fn(pltf=None, use_tex=None, tex_hphantom=None):


    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return misc_fns.calc_tex_hphantom_fn(pltf=pltf, use_tex=use_tex, tex_hphantom=tex_hphantom)


def calc_tex_hphantom_ps(tex_hphantom=None, app_tex_hphantom=None, pltf=None, use_tex=None):

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return misc_fns.calc_tex_hphantom_ps(tex_hphantom=tex_hphantom, app_tex_hphantom=app_tex_hphantom, pltf=pltf, use_tex=use_tex)



def capp_tex_vect(S=None, tex_vect=None, app_tex_vect=None, app_null=None, reshape=None):
    if app_null is None:
        app_null = False
    return misc_fns.app_presuff(S=S, ps=calc_tex_vect_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_vect(S=None, tex_vect=None, app_tex_vect=None, reshape=None):
    return misc_fns.check_presuff(S=S, ps=calc_tex_vect_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_vect(S=None, tex_vect=None, app_tex_vect=None, reshape=None):
    return misc_fns.rm_presuff(S=S, ps=calc_tex_vect_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_tex_num(S=None, tex_num=None, app_tex_num=None, app_null=None, reshape=None, pltf=None, use_tex=None):
    if app_null is None:
        app_null = False
    return misc_fns.app_presuff(S=S, ps=calc_tex_num_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_num(S=None, tex_num=None, app_tex_num=None, reshape=None, pltf=None, use_tex=None):
    return misc_fns.check_presuff(S=S, ps=calc_tex_num_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_num(S=None, tex_num=None, app_tex_num=None, reshape=None, pltf=None, use_tex=None):
    return misc_fns.rm_presuff(S=S, ps=calc_tex_num_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, app_null=None, reshape=None, pltf=None, use_tex=None):
    if app_null is None:
        app_null = False
    return misc_fns.app_presuff(S=S, ps=calc_tex_hphantom_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, reshape=None, pltf=None, use_tex=None):
    return misc_fns.check_presuff(S=S, ps=calc_tex_hphantom_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, reshape=None, pltf=None, use_tex=None):
    return misc_fns.rm_presuff(S=S, ps=calc_tex_hphantom_ps(**misc_fns.dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


#### hawkes

def get_matplt_use_tex():

    return matplotlib.rcParams["text.usetex"]


def calc_tex_preamble_print_lines():

    base_tex_preamble_print_lines = [f"\\usepackage{{siunitx}}"]
    return np.concatenate([
                            base_tex_preamble_print_lines,
                            GR.calc_greek_roman_tex_preamble_print_lines(app_omicron=True, def_Greek=True)
                            ])


def calc_tex_preamble():


    return misc_fns.str_concatenate(calc_tex_preamble_print_lines(), fill_str="\n")



def plt_setup(matplt, font_family=None, font_text=None, font_mathtext=None, use_tex=None, dpi=None, SMALL_SIZE=None, MEDIUM_SIZE=None, BIGGER_SIZE=None, style=None, supress_warnings=None):

    if use_tex is None:
        use_tex = False

    if dpi is None:
        dpi=300

    if SMALL_SIZE is None:
        SMALL_SIZE = 8
    if MEDIUM_SIZE is None:
        MEDIUM_SIZE = 9
    if BIGGER_SIZE is None:
        BIGGER_SIZE = 11

    if style is not None:
        matplt.style.use(style)



    # Font
    matplt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    matplt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    matplt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    matplt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    matplt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    matplt.rc('figure', figsize=[3.4, 3])


    if use_tex:

        rc_fonts = {
                        "text.usetex": True,
                        #'text.latex.preview': True, # Gives correct legend alignment.
                        #'mathtext.default': 'regular',
                        'text.latex.preamble': calc_tex_preamble(),
                    }
        matplt.rcParams.update(rc_fonts)

    
    matplt.rcParams['figure.dpi'] = dpi



    if font_family is None:
        font_family = 'serif'

    matplt.rc('font', family=font_family)

    if font_text is None:
        font_text = matplt.rcParams[f"font.{font_family}"]
    matplt.rcParams[f"font.{font_family}"] = misc_fns.make_iterable_array(font_text, as_list=True)

    if font_mathtext is None:
        #font_mathtext = matplt.rcParams[f"font.{font_family}"][0]
        if font_family=="sans-serif":
            font_mathtext = "dejavusans"
        else:
            font_mathtext = "dejavuserif"
            #font_mathtext = "cm"

    matplt.rc('mathtext', fontset=font_mathtext)
    
    pd.plotting.register_matplotlib_converters()

    if supress_warnings is None:
        supress_warnings = True

    if supress_warnings:
        warnings.filterwarnings(
        action='ignore', module='matplotlib.figure', category=UserWarning,
        message=('This figure includes Axes that are not compatible with tight_layout, '
                'so results might be incorrect.')
        )


def rc_plot_setup(name="fig", nrows=1, ncols=1, borderlabels=None, tick_direction=None):

    if borderlabels is None:
        borderlabels = 'leftbottom'


    if tick_direction is None:
        tick_direction = "out"

    fig = {
                "name": name,
                "fig": None,
                "gs": None,
                "ax":   [
                            [None for j in range(ncols)] for i in range(nrows)
                        ],
            }
    fig["fig"] = plt.figure(constrained_layout=True)
    fig["gs"] = fig["fig"].add_gridspec(nrows, ncols)
    for i in range(len(fig["ax"])):
        for j in range(len(fig["ax"][i])):
            fig["ax"][i][j] = fig["fig"].add_subplot(fig["gs"][i, j])


    for i in range(len(fig["ax"])):
        for j in range(len(fig["ax"][i])):
            fig["ax"][i][j].tick_params(
                                            which='both',      
                                            bottom=True, labelbottom=False,     
                                            top=True, labeltop=False,
                                            left=True, labelleft=False,     
                                            right=True, labelright=False,         
                                            direction=tick_direction,
                                        )
            

    if borderlabels=='edge':
        for i in range(len(fig["ax"])):
            for j in range(len(fig["ax"][i])):
                if i==0:
                    if len(fig["ax"]) > 1:
                        fig["ax"][i][j].xaxis.set_label_position("top")
                        fig["ax"][i][j].tick_params(labeltop=True)
                if i==len(fig["ax"])-1:
                    fig["ax"][i][j].tick_params(labelbottom=True)
            if len(fig["ax"][i]) > 1:
                fig["ax"][i][-1].yaxis.set_label_position("right")
                fig["ax"][i][-1].tick_params(labelright=True)
            fig["ax"][i][0].tick_params(labelleft=True)
    elif borderlabels=='leftbottom':
        for i in range(len(fig["ax"])):
            for j in range(len(fig["ax"][i])):
                if i==len(fig["ax"])-1:
                    fig["ax"][i][j].tick_params(labelbottom=True)
            fig["ax"][i][0].tick_params(labelleft=True)
    elif borderlabels=='leftbottomedge':
        for i in range(len(fig["ax"])):
            for j in range(len(fig["ax"][i])):
                if i==len(fig["ax"])-1:
                    fig["ax"][i][j].tick_params(labelbottom=True)
                fig["ax"][i][j].tick_params(labelleft=True)
    elif borderlabels=='leftedgebottomedge':
        for i in range(len(fig["ax"])):
            for j in range(len(fig["ax"][i])):
                fig["ax"][i][j].tick_params(labelbottom=True)
            fig["ax"][i][0].tick_params(labelleft=True)
    else:
        for i in range(len(fig["ax"])):
            for j in range(len(fig["ax"][i])):
                fig["ax"][i][j].tick_params(labelleft=True, labelbottom=True)
    

    fig["fig"].set_size_inches(7, 4)
    fig["fig"].tight_layout()

    return fig


def fig_row_setup(name="fig", wratio=np.array([1,1]), labelleftright=None, tick_direction=None):

    if labelleftright is None:
        labelleftright = True
    if tick_direction is None:
        tick_direction = "out"

    fig = {
                "name": name,
                "fig": None,
                "gs": None,
                "ax":   [
                           [None for j in range(len(wratio))]
                        ],
            }
    fig["fig"] = plt.figure(constrained_layout=True)
    fig["gs"] = fig["fig"].add_gridspec(1, np.sum(np.array(wratio)))

    fig["ax"][0][0] = fig["fig"].add_subplot(fig["gs"][0, :wratio[0]])
    if len(wratio) > 2:
        for j in range(1, len(wratio)-1):
            fig["ax"][0][j] = fig["fig"].add_subplot(fig["gs"][0, np.sum(wratio[:j]):np.sum(wratio[:1+j])])
    fig["ax"][0][-1] = fig["fig"].add_subplot(fig["gs"][0, np.sum(wratio[:-1]):])

    if labelleftright:
        for j in range(len(wratio)):
            fig["ax"][0][j].tick_params(
                                            which='both',      
                                            bottom=True, labelbottom=True,     
                                            top=True, labeltop=False,
                                            left=True, labelleft=False,     
                                            right=True, labelright=False,         
                                            direction=tick_direction,
                                        )
            fig["ax"][0][-1].yaxis.set_label_position('right') 
            fig["ax"][0][-1].tick_params(labelright=True)
            fig["ax"][0][0].tick_params(labelleft=True)
    else:
        for j in range(len(wratio)):
            fig["ax"][0][j].tick_params(
                                            which='both',      
                                            bottom=True, labelbottom=True,     
                                            top=True, labeltop=False,
                                            left=True, labelleft=True,     
                                            right=True, labelright=False,         
                                            direction=tick_direction,
                                      )
    fig["fig"].set_size_inches(7, 4)
    fig["fig"].tight_layout()

    return fig



def fig_column_setup(name="fig", hratio=np.array([2,1]), labeltopbottom=None, tick_direction=None):

    if labeltopbottom is None:
        labeltopbottom = True
    if tick_direction is None:
        tick_direction = "out"

    fig = {
                "name": name,
                "fig": None,
                "gs": None,
                "ax":   [
                            [None] for i in range(len(hratio))
                        ],
            }
    fig["fig"] = plt.figure(constrained_layout=True)
    fig["gs"] = fig["fig"].add_gridspec(np.sum(np.array(hratio)), 1)

    fig["ax"][0][0] = fig["fig"].add_subplot(fig["gs"][:hratio[0], 0])
    if len(hratio) > 2:
        for i in range(1, len(hratio)-1):
            fig["ax"][i][0] = fig["fig"].add_subplot(fig["gs"][np.sum(hratio[:i]):np.sum(hratio[:1+i]), 0])
    fig["ax"][-1][0] = fig["fig"].add_subplot(fig["gs"][np.sum(hratio[:-1]):, 0])

    if labeltopbottom:
        for i in range(len(hratio)):
            fig["ax"][i][0].tick_params(
                                            which='both',      
                                            bottom=True, labelbottom=False,     
                                            top=True, labeltop=False,
                                            left=True, labelleft=True,     
                                            right=True, labelright=False,         
                                            direction=tick_direction,
                                        )
            fig["ax"][0][0].xaxis.set_label_position('top') 
            fig["ax"][0][0].tick_params(labeltop=True)
            fig["ax"][-1][0].tick_params(labelbottom=True)
    else:
        for i in range(len(hratio)):
            fig["ax"][i][0].tick_params(
                                            which='both',      
                                            bottom=True, labelbottom=True,     
                                            top=True, labeltop=False,
                                            left=True, labelleft=True,     
                                            right=True, labelright=False,         
                                            direction=tick_direction,
                                      )
    fig["fig"].set_size_inches(7, 4)
    fig["fig"].tight_layout()

    return fig

def copy_line_ax(ax, line, yscale=None):

    [x,y] = line.get_data()
    if yscale is not None:
        ax.plot(x, yscale*y, color=line.get_color(), linewidth=line.get_linewidth(), alpha=line.get_alpha())
    else:
        ax.plot(x, y, color=line.get_color(), linewidth=line.get_linewidth(), alpha=line.get_alpha())
    return ax

def copy_rect_ax(ax, rect):

    rect2 = plt.Rectangle(xy=rect.get_xy(), width=rect.get_width(), height=rect.get_height(),
                                    facecolor=rect.get_facecolor(), alpha=rect.get_alpha(),
                                    #edgecolor=rect.get_edgecolor(), linewidth=rect.get_linewidth())
                                    edgecolor=None, linewidth=None)
                                    

    ax.add_patch(rect2)
    return ax

def copy_ax_properties(ax, ax_old):

    ax.set_xlim(ax_old.get_xlim())
    ax.set_ylim(ax_old.get_ylim())
    ax.set_xlabel(ax_old.get_xlabel())
    ax.set_ylabel(ax_old.get_ylabel())


    return ax

def calc_cols(flip_01=None):

    if flip_01 is None:
        flip_01 = False

    cols = np.array(['#12476d', '#ff7f0e', '#1c661c', '#ff7878', '#9780c4', '#bf3c30', '#ccb833', '#bfbfbf', '#70960f', '#15b0c1'])

    if False:
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

        cols[0] = '#12476d'
        #cols[1] = '#ff7f0e'
        cols[2] = '#1c661c'
        #cols[3] = '#ff7373'
        cols[3] = '#ff7878'
        #cols[4] = '#8b60b3'
        cols[4] = '#9780c4'
        cols[5] = '#bf3c30'
        cols[6] = '#ccb833'
        cols[7] = '#bfbfbf'
        cols[8] = '#70960f'
        #cols[9] = '#14a3b3'
        cols[9] = '#15b0c1'



    if False:
        cols[3] = '#ff7878'

        #cols[6] = '#e788ca'
        #cols[4] = '#9f78c4'
        #cols[4] = '#a27cc5'
        #cols[4] = '#a47fc7'


        cols[6] = '#ccb833'
        cols[5] = '#bf3c30'
        #cols[4] = '#947cc4'
        cols[4] = '#9780c4'

    if flip_01:
        cols_r = [cols[1-i] for i in range(2)]
        for i in range(2):
            cols[i] = cols_r[i]

    return np.array(cols)


def calc_cols_tail(rows=None):

    if rows is None:
        rows = False

    cols = calc_cols(flip_01=False)

    cols_tail = [
                    [cols[1], cols[0], cols[8]],
                    [cols[3], cols[2], cols[9]],
                    [cols[6], cols[5], cols[4]],
                    [cols[7]]
                ]


    if rows:
        return cols_tail
    else:
        return np.concatenate(cols_tail)
    

def calc_cols_tail_grad(group=None):

    if group is None:
        group=False


    cols_tail = calc_cols_tail(rows=True)

    if group:
        cols_tail_grad = [cols_tail[k//3][(1+k)%3] for k in range(9)]
    else:
        cols_tail_grad = [cols_tail[k%3][(1+(k//3))%3] for k in range(9)]

    cols_tail_grad.append(cols_tail[3][0])

    return cols_tail_grad



def calc_cols_CI(N_CI=None, bound_cols=None, high_to_low=None):

    if high_to_low is None:
        high_to_low = False

    if bound_cols is None:
        bound_cols = ['#ebebeb', '#d7d7d7']
        if high_to_low:
            bound_cols = np.flip(bound_cols)
    if N_CI is None:
        N_CI = 2

    if N_CI<2:
        cols_CI = [bound_cols[0]]
    elif N_CI==2:
        cols_CI = bound_cols
    else:
        bound_cols_rgb = np.array([np.array(hex_to_rgb(value=bound_cols[k])) for k in range(2)])
        iter_rgb = np.diff(bound_cols_rgb, axis=0)[0,:]/(N_CI-1)
        cols_CI = [rgb_to_hex(rgb=bound_cols_rgb[0,:] + i*iter_rgb) for i in range(N_CI)]

    return cols_CI


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):

    rgb = np.array(rgb, dtype=int)
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def colour_transform_alpha(value_foreground, value_background="#ffffff", alpha=1):
    rgb_foreground = np.array(hex_to_rgb(value_foreground))
    rgb_background = np.array(hex_to_rgb(value_background))

    rgb_output = rgb_background + (rgb_foreground - rgb_background)*alpha

    rgb_output_tuple = tuple([int(rgb_output[i]) for i in range(3)])

    value_output = rgb_to_hex(rgb_output_tuple)

    return value_output

def buffer(b=None, x=None):

    if x is None:
        x = np.array([0,1])
    if b is None:
        b = 1.0

    mid = (x[1]+x[0])/2
    diff = (x[1]-x[0])/2

    return mid + b*np.array([-1,1])*diff

def autocorrelation_TS_old(data, window=50, lag=1):
    return np.array([np.corrcoef(data[lag+i:lag+window+i], data[i:window+i])[0,1] for i in range(1+data.size-window-lag)])

def correlogram_old(data, lags=range(1,50)):

    lags = misc_fns.make_iterable_array(lags)

    return np.array([np.corrcoef(data[i:], data[:-i])[1,0] for i in lags])

def autocorrelation_TS(data, window=None, lag=None):

    if window is None:
        window = 50

    if lag is None:
        lag = 1

    #return np.array([np.corrcoef(data[lag+i:lag+window+i], data[i:window+i])[0,1] for i in range(1+data.size-window-lag)])

    return np.array([acf(data[i:lag+window+i], nlags=lag, qstat=False, adjusted=False)[-1] for i in range(1+data.size-(window+lag))])

def correlogram(data, lags=None):

    if lags is None:
        lags = np.arange(1,50)
    lags = misc_fns.make_iterable_array(lags)

    max_lag = np.max(lags)

    #return np.array([np.corrcoef(data[i:], data[:-i])[1,0] for i in lags])

    return acf(data, nlags=max_lag, qstat=False, adjusted=False)[lags]





def calc_C_str(C_str_nl_pre=None, C_str_matter=None, C_str_n=None, C_str_l=None, C_str_n_post=None, C_str_l_post=None, C_str_initchr=None):

    if C_str_nl_pre is None:
        C_str_nl_pre = "C"

    if C_str_matter is None:
        C_str_matter = "m"

    if C_str_n_post is None:
        C_str_n_post = f''

    if C_str_l_post is None:
        C_str_l_post = f''

    if C_str_initchr is None:
        C_str_initchr = "A"



    if C_str_n is None:
        C_str_n = ''
        if C_str_l is not None:
            if type(C_str_l)==str and len(C_str_l)==1:
                C_str_n = 1+ord(C_str_l)-ord(C_str_initchr)
            

    if C_str_l is None:
        C_str_l = ''
        if type(C_str_n)==int:
            if C_str_n>0:
                C_str_l = chr(C_str_n+ord(C_str_initchr)-1)

            
    C_str_nl = {'n': C_str_n, 'l': C_str_l}
    C_str_nl_post = {'n': C_str_n_post, 'l': C_str_l_post}

    C_str = {nl: f"{C_str_nl_pre}{C_str_matter}{C_str_nl[nl]}{C_str_nl_post[nl]}" for nl in C_str_nl}


    return C_str



def calc_an_params(an_mode=None, an_xycoords=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None):

    if an_xycoords is None:
        an_xycoords = 'axes fraction'

    if an_fs is None:
        if an_mode=='thesis':
            an_fs = 7
        else:
            an_fs = 8

    if an_x0 is None:
        an_x0 = 0.02
    if an_y0 is None:
        an_y0 = 0.02
    if an_xi is None:
        if an_mode=='thesis':
            an_xi = 0.07
        else:
            an_xi = 0.08
    if an_yi is None:
        if an_mode=='thesis':
            an_yi = 0.07
        else:
            an_yi = 0.08

    an_params = {
                    'an_mode': an_mode,
                    'an_xycoords': an_xycoords,
                    'an_fs': an_fs,
                    'an_x0': an_x0,
                    'an_y0': an_y0,
                    'an_xi': an_xi,
                    'an_yi': an_yi,
                }
    
    return an_params


def calc_an_params_args(cat=None, cat_rm=None, list_rm=None):

    args =  {
                'main': ['an_mode', 'an_xycoords', 'an_fs', 'an_x0', 'an_y0', 'an_xi', 'an_yi'],
                'ow': ['an_fs_ow', 'an_x0_ow', 'an_y0_ow', 'an_xi_ow', 'an_yi_ow'],
                'dir': ['an_h', 'an_v', 'an_xs', 'an_ys'],
                'flip': ['flip_x0ry0t']
            }
    
    return misc_fns.return_args(args=args, cat=cat, cat_rm=cat_rm, list_rm=list_rm)
    



    

def calc_an_params_dir(an_mode=None, an_xycoords=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, 
                       an_fs_ow=None, an_x0_ow=None, an_y0_ow=None, an_xi_ow=None, an_yi_ow=None,
                       an_h=None, an_v=None, an_xs=None, an_ys=None,
                       flip_x0ry0t=None):
    
    if flip_x0ry0t is None:
        flip_x0ry0t = False

    if an_h is None:
        an_h = 'left'

    if an_v is None:
        an_v = 'bottom'
        
    if an_xs is None:
        if an_v=='right':
            an_xs = -1
        else:
            an_xs = 1

    if an_ys is None:
        if an_v=='top':
            an_ys = -1
        else:
            an_ys = 1

    _locals = locals()
    an_params = calc_an_params(**{c: _locals[c] for c in calc_an_params_args(cat='main')})

    if an_fs_ow is not None:
        an_params['an_fs'] = an_fs_ow
    if an_x0_ow is not None:
        an_params['an_x0'] = an_x0_ow
    if an_y0_ow is not None:
        an_params['an_y0'] = an_y0_ow
    if an_xi_ow is not None:
        an_params['an_xi'] = an_xi_ow
    if an_yi_ow is not None:
        an_params['an_yi'] = an_yi_ow

    an_params['an_h'] = an_h
    an_params['an_v'] = an_v
    an_params['an_xs'] = an_xs
    an_params['an_ys'] = an_ys

    if flip_x0ry0t:
        if an_params['an_h']=='right':
            an_params['an_x0'] = 1 - an_params['an_x0']

        if an_params['an_v']=='top':
            an_params['an_y0'] = 1 - an_params['an_y0']

    return an_params

    


def annotate_array(str_list=None, col_list=None,
                   an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None, flip_x0ry0t=None,
                   an_xM=None, an_yM=None,
                   top_to_bottom=None, left_to_right=None, flip_i=None, flip_j=None,
                   ax=None,
                  ):
    
    _locals = locals()
    an_params = calc_an_params_dir(**{c: _locals[c] for c in calc_an_params_args(cat=['main', 'dir', 'flip'])})

    if top_to_bottom is None:
        top_to_bottom = False
    if left_to_right is None:
        left_to_right = False

    if flip_i is None:
        if top_to_bottom:
            if an_params['an_ys']>0:
                flip_i = True

    if flip_j is None:
        if left_to_right:
            if an_params['an_xs']<0:
                flip_j = True

    

    if str_list is None:
        str_list = []

    str_list = misc_fns.list_2D(val=str_list, set_J_max=True, mode_1D=None, flip_i=flip_i, flip_j=flip_j)

    I_max = len(str_list)
    J_max = [len(str_list[i]) for i in range(I_max)]
    J_max_eff = 0
    if len(J_max)>0:
        J_max_eff = np.max(J_max)

    xy_box = calc_xy_box(an_params=an_params, P_y=I_max, P_x=J_max_eff)


    if col_list is None:
        col_list="black"
    col_list = misc_fns.list_2D(val=col_list, I_max=I_max, J_max=J_max, mode_1D=None, flip_i=flip_i, flip_j=flip_j)

    if an_yM is None:
        an_yM = np.arange(I_max)
    an_yM = misc_fns.make_iterable_array(an_yM)

    if an_xM is None:
        an_xM = [np.arange(J_max[i]) for i in range(I_max)]
    an_xM = misc_fns.list_2D(val=an_xM, I_max=I_max, J_max=J_max, mode_1D=None)


    

    if ax is not None:
        for i in range(I_max):
            for j in range(J_max[i]):

                ax.annotate(str_list[i][j], 
                                                xy=(an_params['an_x0']+an_params['an_xs']*an_xM[i%I_max][j%J_max[i]]*an_params['an_xi'], an_params['an_y0']+an_params['an_ys']*an_yM[i%I_max]*an_params['an_yi']), xycoords=an_params['an_xycoords'],
                                                horizontalalignment=an_params['an_h'],
                                                verticalalignment=an_params['an_v'],
                                                fontsize=an_params['an_fs'],
                                                color=col_list[i][j])
            



    return xy_box


def annotate_pvals(
                   p_vals, p_cols=None, p_symb=None,
                   p_thresh=None, N_p_thresh=None,
                   dp=None,
                   an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None, flip_x0ry0t=None,
                   top_to_bottom=None, left_to_right=None, flip_i=None, flip_j=None,
                   brac_type=None, add_lr=None, null_brac_space=None,
                   ax=None,
                   ):
        

    _locals = locals()
    an_params = calc_an_params_dir(**{c: _locals[c] for c in calc_an_params_args(cat=['main', 'dir', 'flip'])}) 

    if top_to_bottom is None:
        top_to_bottom = True
    
    if dp is None:
        dp = 1


    P = np.size(p_vals)

    if p_cols is None:
        p_cols = ['black' for p in range(P)]
    if p_symb is  None:
        p_symb = ['p' for p in range(P)]
    if p_thresh is None:
        p_thresh = [0.05, 0.01]
    p_thresh = np.flip(np.sort(misc_fns.make_iterable_array(p_thresh)))

    if N_p_thresh is None:
        N_p_thresh = np.size(p_thresh)

    p_str = [None for p in range(P)]

    if brac_type is None:
        brac_type = '[]'
    if add_lr is None:
        add_lr = False

    if null_brac_space is None:
        null_brac_space = misc_fns.str_concatenate(np.repeat("\\:", 2))

    for p in range(P):
        val_str = misc_fns.scif_string(p_vals[p], se=None, dp=dp, app_dlrs=False, app_tex_num=False)

        if N_p_thresh==0:
            KS_str = val_str
        else:
            p_thresh_less = p_vals[p] < p_thresh[:N_p_thresh]
            N_p_thresh_less = np.sum(p_thresh_less)
            N_p_thresh_more = N_p_thresh - N_p_thresh_less

            if True:
                KS_str = misc_fns.app_presuff_i(S=misc_fns.capp_brac(val_str, brac_type=brac_type, add_lr=add_lr, repeat=N_p_thresh_less), 
                                                        ps=np.repeat(misc_fns.str_concatenate(np.repeat(null_brac_space, N_p_thresh_more)),2), 
                                                        app_null=True)
            else:
                KS_str_more = misc_fns.str_concatenate(np.repeat(f'\\:\\:', N_p_thresh_more))
                KS_str = f"{KS_str_more}{misc_fns.capp_brac(val_str, brac_type='[]', add_lr=False, repeat=N_p_thresh_less)}"



        p_str[p] = f"${p_symb[p]} = {KS_str}$"

    
    xy_box = calc_xy_box(an_params=an_params, P_y=P, along_x=False, along_y=True)
    
    if ax is not None:
        annotate_array(ax=ax, str_list=p_str,
                                **an_params,
                                top_to_bottom=top_to_bottom, left_to_right=left_to_right, flip_i=flip_i, flip_j=flip_j,
                                col_list=p_cols)
        
    return xy_box
            


def calc_a_box(an_params=None, a=None,
                    a_0=None, a_1=None, a_0_1=None, P_a=None, along_a=None, sort=None, default_full=None):
    
    if a is None:
        a = 'y'

    if along_a is None:
        along_a = True

    if sort is None:
        sort = True

    if default_full is None:
        default_full = False

    a_box = np.zeros(2, dtype=float)

    if an_params is None:
        an_params = {val: 0.0 for val in calc_an_params_args(cat=['main', 'dir'])}

    if a_0 is None:
        a_0 = an_params[f'an_{a}0']


    if a_1 is None:
        if a_0_1 is None:
            if P_a is None:
                P_a = 0

            if P_a>0:
                a_0_1 = an_params[f'an_{a}s']*(P_a)*an_params[f'an_{a}i']
            else:
                if default_full:
                     a_0_1 = 1 - a_0
                else:
                     a_0_1 = 0.0
               

        a_1 = a_0 + a_0_1


    
    a_box[0] = a_0
    if along_a:
        a_box[1] = a_1
    else:
        a_box[1] = a_box[0]
    

    if sort:
        a_box = np.sort(a_box)

    return a_box




def calc_a_f_boxes(a_boxes, along_a_bool=None, a_rm_zw=None):

    if a_rm_zw is None:
        a_rm_zw = True


    if along_a_bool is None:
        along_a_bool = np.full(np.shape(a_boxes)[0], True)

    if a_rm_zw:
        along_a_bool[np.diff(a_boxes, axis=-1)[...,0]==0] = False

    a_s_boxes = np.sort(a_boxes[along_a_bool,:], axis=-1)

    if np.any(along_a_bool):

        overlap = np.logical_and(np.tril(np.greater_equal.outer(a_s_boxes[:,0], a_s_boxes[:,0])), np.tril(np.less_equal.outer(a_s_boxes[:,0], a_s_boxes[:,1])))
        where_overlap = np.where(overlap)
        where_overlap_key = {i: np.unique(np.concatenate([where_overlap[1][where_overlap[0]==i], where_overlap[0][where_overlap[1]==i]])) for i in range(np.shape(a_s_boxes)[0])}

        a_f_boxes = np.array([[np.min(a_s_boxes[where_overlap_key[i],0]), np.max(a_s_boxes[where_overlap_key[i],1])] for i in where_overlap_key])
        a_f_boxes = a_f_boxes[np.argsort(a_f_boxes[:,0]),:]
        a_f_boxes = a_f_boxes[np.concatenate([[True], np.all(np.diff(a_f_boxes, axis=0)!=0, axis=1)])]

    else:
        a_f_boxes = a_s_boxes

    return a_f_boxes




def calc_between_a_boxes(a_boxes, a_box=None, along_a=None, along_a_bool=None, a_rm_zw=None):


    a_boxes = np.array(a_boxes)


    if along_a is None:
        along_a = True

    

    if a_box is None:
        a_box = calc_a_box(default_full=True)

    if along_a:

        a_f_boxes = calc_a_f_boxes(a_boxes=a_boxes, along_a_bool=along_a_bool, a_rm_zw=a_rm_zw)

        if np.shape(a_f_boxes)[0]>0:
            a_bds = np.concatenate([
                                            [a_box[0]],
                                            np.concatenate(a_f_boxes),
                                            [a_box[1]],
                                            ])
            
            N_along_a = np.shape(a_f_boxes)[0]
            
            a_max_gap_i = np.argmax(np.diff(a_bds)[np.arange(1+2*N_along_a)%2==0])
            if a_max_gap_i<(N_along_a):
                a_box[1] = a_f_boxes[a_max_gap_i,0]

            if a_max_gap_i>0:
                a_box[0] = a_f_boxes[a_max_gap_i-1,1]


    return a_box
    


def calc_xy_box(an_params=None, 
               x_0=None, x_1=None, y_0=None, y_1=None, x_0_1=None, y_0_1=None,
               P_x=None, P_y=None,
               along_x=None, along_y=None, sort=None, default_full=None):
    
    _locals = locals()
    if an_params is None:
        an_params = {val: 0.0 for val in calc_an_params_args(cat=['main', 'dir'])}

    a_s = ['x', 'y']
    a_s_args = {a: {
                        'a': a,
                        **{f"a_{arg}": _locals[f"{a}_{arg}"] for arg in ['0', '1', '0_1']},
                        **{f"{arg}_a": _locals[f"{arg}_{a}"] for arg in ['P', 'along']},

                    } for a in a_s}


    return np.array([calc_a_box(an_params=an_params, sort=sort, default_full=default_full, **a_s_args[a]) for a in a_s])

    




def calc_between_xy_boxes(xy_boxes, xy_box=None, along_x=None, along_y=None, along_x_bool=None, x_rm_zw=None, along_y_bool=None, y_rm_zw=None):


    xy_boxes = np.array(xy_boxes)


    if xy_box is None:
        xy_box = calc_xy_box(default_full=True)


    
    return np.array([
                        calc_between_a_boxes(a_boxes=xy_boxes[:,0,:], a_box=xy_box[0,:], along_a=along_x, along_a_bool=along_x_bool, a_rm_zw=x_rm_zw),
                        calc_between_a_boxes(a_boxes=xy_boxes[:,1,:], a_box=xy_box[1,:], along_a=along_y, along_a_bool=along_y_bool, a_rm_zw=y_rm_zw),
                    ])



def calc_bbox(xy_box=None, xy_boxes=None, along_x=None, along_y=None, along_x_bool=None, along_y_bool=None):

    if xy_box is None:
        xy_box = calc_xy_box(default_full=True)

    if xy_boxes is not None:
        xy_box = calc_between_xy_boxes(xy_boxes=xy_boxes, xy_box=xy_box, along_x=along_x, along_y=along_y, along_x_bool=along_x_bool, along_y_bool=along_y_bool)

    return (
            xy_box[0,0],
            xy_box[1,0],
            xy_box[0,1]-xy_box[0,0],
            xy_box[1,1]-xy_box[1,0],
        )
    

def calc_table_head(heads=None, lead=None, pre=None, post=None, c0=None, cw=None, calign=None, q_same_row=None, q_space=None, q_above_empty_lead=None, last_head=None, N_p_max=None, p_same_rows=None, p_space=None, repeat=None, r_space=None, incl_hline=None, incl_cmidrules=None):

    if last_head is None:
        last_head = False
    


    if incl_hline is None:
        if last_head:
            incl_hline = True
        else:
            incl_hline = False
    if incl_cmidrules is None:
        if last_head:
            incl_cmidrules = False
        else:
            incl_cmidrules = True

    if p_space is None:
        if last_head:
            p_space = " & "
        else:
            p_space = " & & "
    

    if repeat is None:
        repeat = 1

    if r_space is None:
        r_space = " & & "

    


    heads = np.array(heads)
    if len(np.shape(heads))==1:
        heads = np.transpose([heads])
    heads_shape = np.array(np.shape(heads))

        
    N_p = heads_shape[0]
    N_q = heads_shape[1]

    
    if lead is None:
        lead = ''
    lead = misc_fns.make_iterable_array(lead)
    N_lead = np.size(lead)

    if pre is None:
        pre = ''
    if post is None:
        post = ''

    
    if c0 is None:
        c0 = [lead[k].count("&") for k in range(lead.size)]
    c0 = misc_fns.make_iterable_array(c0)
    N_c0 = np.size(c0)

    

    

    if cw is None:
        cw = 1

    if calign is None:
        calign = 'c'


    if q_same_row is None:
        q_same_row = True
        
    if p_same_rows is None:
        p_same_rows = True

    if N_p_max is None:
        if p_same_rows:
            N_p_max = N_p
        else:
            N_p_max = 1

    N_rows = 1+((N_p-1)//N_p_max)

    P_rows = np.repeat(N_p_max, N_rows)
    P_rows[-1] = 1 + ((N_p-1) % N_p_max)


    if q_space is None:
        q_space = " "

    if q_above_empty_lead is None:
        q_above_empty_lead = True

    if q_above_empty_lead:
        empty_lead = [misc_fns.str_concatenate(["& " for k in range(c0[i])]) for i in range(N_c0)]

    
    p_space_c = p_space.count('&') - 1
    r_space_c = r_space.count('&') - 1
    

    cp = np.array([P_rows[i]*(p_space_c+cw)-p_space_c for i in range(len(P_rows))])
        

    row_strs = ['' for i in range(N_rows)]
    for i in range(N_rows):
        if not q_same_row:
            for q in range(N_q):
                if q!=0:
                    row_strs[i] += '\n'
                    pre_eff = ''
                    post_eff = ''
                else:
                    pre_eff = pre
                    post_eff = post
                if q_above_empty_lead and q<N_q-1:
                    row_strs[i] += empty_lead[q%N_lead]
                else:
                    row_strs[i] += lead[q%N_lead]
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([misc_fns.str_concatenate([f'\\multicolumn{{{cw}}}{{{calign}}}{{{pre_eff}{misc_fns.str_concatenate([heads[i*N_p_max+p][q]], fill_str=q_space, fill_str_void=-1)}{post_eff}}}' for p in range(P_rows[i])], fill_str=p_space, fill_str_void=-1) for r in range(repeat)], fill_str=r_space, fill_str_void=-1)
                row_strs[i] += '\n'
                row_strs[i] += '\\\\'
            if incl_cmidrules:
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([misc_fns.str_concatenate([f"\\cmidrule{{{1+c0[i%N_c0]+r*(r_space_c+cp[i])+p*(p_space_c+cw)}-{c0[i%N_c0]+r*(r_space_c+cp[i])+(1+p)*(p_space_c+cw)-(p_space_c)}}}" for p in range(P_rows[i])]) for r in range(repeat)])
            if incl_hline:
                row_strs[i] += '\n'
                row_strs[i] += f"\\hline"

        else:
            row_strs[i] += lead[i%N_lead]
            row_strs[i] += '\n'
            row_strs[i] += misc_fns.str_concatenate([misc_fns.str_concatenate([f'\\multicolumn{{{cw}}}{{{calign}}}{{{pre}{misc_fns.str_concatenate([heads[i*N_p_max+p][q] for q in range(N_q)], fill_str=q_space, fill_str_void=-1)}{post}}}' for p in range(P_rows[i])], fill_str=p_space, fill_str_void=-1) for r in range(repeat)], fill_str=r_space, fill_str_void=-1)
            row_strs[i] += '\n'
            row_strs[i] += '\\\\'
            if incl_cmidrules:
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([misc_fns.str_concatenate([f"\\cmidrule{{{1+c0[i%N_c0]+r*(r_space_c+cp[i])+p*(p_space_c+cw)}-{c0[i%N_c0]+r*(r_space_c+cp[i])+(1+p)*(p_space_c+cw)-(p_space_c)}}}" for p in range(P_rows[i])]) for r in range(repeat)])
            if incl_hline:
                row_strs[i] += '\n'
                row_strs[i] += f"\\hline"

    return row_strs

def calc_cmode_dict(cmode=None, N_cap=None, 
                    lead=None,
                    pre=None, post=None, calign=None, embed_head=None,
                    head=None, ticks=None, periods=None, tails=None,
                    N_head=None, N_tick=None, N_period=None, N_tail=None, 
                    heads_head=None, heads_tick=None, heads_period=None, heads_tail=None):
    
    

    if lead is None:
        lead = ''

    c_lead = lead.count('&')
    empty_lead = misc_fns.str_concatenate(['& ' for c in range(c_lead)])


    if N_cap is None:
        N_cap = 12

    if pre is None:
        pre = {}
    if post is None:
        post = {}
    

    if head is None:
        head = ""
    if ticks is None:
        ticks = []
    if periods is None:
        periods = []
    if tails is None:
        tails = []

    if N_head is None:
        if len(head)==0:
            N_head = 0
        else:
            N_head = 1
    if N_tick is None:
        N_tick = len(ticks)
    if N_period is None:
        N_period = len(periods)
    if N_tail is None:
        N_tail = len(tails)

    if cmode is None:
        cmode = []
        if N_head>0:
            cmode.append('head')
        if len(ticks)>0:
            cmode.append('tick')
        if len(periods)>0:
            cmode.append('period')
        if len(tails)>0:
            cmode.append('tail')

    if calign is None:
        calign = {}
    elif type(calign)==str:
        calign = {cm: calign for cm in cmode}

    if heads_head is None:
        heads_head = [[f"{head}"]]
    if heads_tick is None:
        heads_tick = [[tick] for tick in ticks]
    if heads_period is None:
        heads_period_dict = {
                                'train': 'In-sample',
                                'forecast': 'Out-of-sample',
                                'all': 'In- and Out-of-sample'

                            }
        heads_period = [[heads_period_dict[period]] for period in periods]
    if heads_tail is None:
        tails_dict = {
                                'left': '\\leftarrow',
                                'right': '\\rightarrow',
                                'both': '\\leftrightarrow'
                            }
        heads_tail = [[tails_dict[tail]] for tail in tails]
        #heads_tail = [[hwks_fns.get_tail_tex(tail)] for tail in tails]



    cmode_dict = {
                        'cmode': cmode,
                        'N_cap': None,
                        'N': dict(),
                        'N_p': dict(),
                        'N_i': dict(),
                        'N_eff': dict(),
                        'cp': dict(),
                        'cw': dict(),
                        'cw_eff': dict(),
                        'repeat': dict(),
                        'p_same_row': dict(),
                        'where_C': None,
                        'C': None, 
                        'heads': {
                                    'head': heads_head,
                                    'tick': heads_tick,
                                    'period': heads_period,
                                    'tail': heads_tail,
                                },
                    }

    cmode_dict['N_cap'] = N_cap

    cmode_dict['N']['dict'] = {
                                'head': N_head,
                                'tick': N_tick,
                                'period': N_period,
                                'tail': N_tail}
    cmode_dict['N']['array'] = np.array([cmode_dict['N']['dict'][l] for l in cmode_dict['cmode']])

    cmode_dict['N_p']['array'] = np.flip(np.cumproduct(np.flip(cmode_dict['N']['array'])))
    cmode_dict['N_p']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['N_p']['array'][l] for l in range(len(cmode_dict['cmode']))}

    cmode_dict['N_i']['array'] = 1+(cmode_dict['N_p']['array']-1)//cmode_dict['N_cap']
    cmode_dict['N_i']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['N_i']['array'][l] for l in range(len(cmode_dict['cmode']))}

    cmode_dict['I'] = np.product(cmode_dict['N_i']['array'])

    cmode_dict['cp']['array'] = np.concatenate([(cmode_dict['N']['array'][-1]+1)*np.flip(np.cumproduct(np.flip(cmode_dict['N']['array'][:-1]))) - 1, [cmode_dict['N']['array'][-1]]])
    cmode_dict['cp']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['cp']['array'][l] for l in range(len(cmode_dict['cmode']))}

    cmode_dict['cw']['array'] = np.concatenate([cmode_dict['cp']['array'][1:], [1]])
    cmode_dict['cw']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['cw']['array'][l] for l in range(len(cmode_dict['cmode']))}



    cmode_dict['p_same_row']['array'] = cmode_dict['N_i']['array']==1
    cmode_dict['p_same_row']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['p_same_row']['array'][l] for l in range(len(cmode))}

    cmode_dict['where_C'] = np.where(cmode_dict['p_same_row']['array'])[0][0]
    cmode_dict['C'] = cmode_dict['cw']['array'][cmode_dict['where_C']]

    cmode_dict['cw_eff']['array'] = np.copy(cmode_dict['cw']['array'])
    cmode_dict['cw_eff']['array'][:cmode_dict['where_C']] = cmode_dict['C']
    cmode_dict['cw_eff']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['cw_eff']['array'][l] for l in range(len(cmode_dict['cmode']))}


    cmode_dict['N_eff']['array'] = np.copy(cmode_dict['N']['array'])
    cmode_dict['N_eff']['array'][:cmode_dict['where_C']] = 1
    cmode_dict['N_eff']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['N_eff']['array'][l] for l in range(len(cmode_dict['cmode']))}

    cmode_dict['repeat']['array'] = np.concatenate([[1], np.cumproduct(cmode_dict['N_eff']['array'])[:-1]])
    cmode_dict['repeat']['dict'] = {cmode_dict['cmode'][l]: cmode_dict['repeat']['array'][l] for l in range(len(cmode_dict['cmode']))}




    cmode_dict['pre'] = {c: '' for c in cmode_dict['cmode']}
    cmode_dict['post'] = {c: '' for c in cmode_dict['cmode']}
    cmode_dict['calign'] = {c: 'c' for c in cmode_dict['cmode']}
    for cm in pre:
        cmode_dict['pre'][cm] = pre[cm]
    for cm in post:
        cmode_dict['post'][cm] = post[cm]
    for cm in calign:
        cmode_dict['calign'][cm] = calign[cm]


    if embed_head is None:
        if not np.isin('head', cmode_dict['cmode']):
            embed_head = True
        else:
            embed_head = False

        
    
    if embed_head and cmode_dict['cmode'][0]!='head':
        cmode_dict['pre'][cmode_dict['cmode'][0]] = f"{misc_fns.str_concatenate(cmode_dict['heads']['head'][0], fill_str=' ', fill_str_void=-1)} $|$ " +  cmode_dict['pre'][cmode_dict['cmode'][0]]

    cmode_dict['lead'] = {c: empty_lead for c in cmode_dict['cmode']}
    cmode_dict['lead'][cmode_dict['cmode'][-1]] = lead

    cmode_dict['head'] = dict()
    H = len(cmode_dict['cmode'])


    for h in range(H):
        cm = cmode_dict['cmode'][h]
        cmode_dict['head'][cm] = calc_table_head(heads=cmode_dict['heads'][cm], lead=cmode_dict['lead'][cm], 
                                                            pre=cmode_dict['pre'][cm], post=cmode_dict['post'][cm], calign=cmode_dict['calign'][cm],
                                                            c0=None, 
                                                            cw=cmode_dict['cw_eff']['dict'][cm], N_p_max=None, q_same_row=False, q_space=None, q_above_empty_lead=None, 
                                                            p_same_rows=cmode_dict['p_same_row']['dict'][cm], repeat=cmode_dict['repeat']['dict'][cm], last_head=(h==H-1))
    

    return cmode_dict