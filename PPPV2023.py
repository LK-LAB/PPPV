#!/usr/bin/env pythonfigure
# --------------------------------------------------------------------
#   ####  PPPV                                          ##### 
#   ####  Primitive Program which Plots band for Vasp   ##### 
#   1/8/2012  Kyoo Kim & Prof. CJ Kang. 
#   1/9/2012  k-scale added
#   1/9/2023  Updated to latest Python by Donggeon Lee
# --------------------------------------------------------------------
import sys,os,re,time,shutil
from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

#-----------FUNCTIONS-----------------------------------------------------------------------------------------
def is_valid_file(filename):
    "check existance of file."
    if (os.path.isfile(filename)==False) or os.path.getsize(filename) == 0:
        return False
    else :
        return True

def input_template(object_file):
     object_file.write("##===========Input for PPPV ===========================================================\n")
     object_file.write("#GENERIC SETUPS                                                                        \n")
     object_file.write("nkpt_intfac    =   3              # number of interpolated kpts                        \n")
     object_file.write("e_min,e_max,de =  -2.0,1.0,0.5    # energy range for plot , tick interval              \n")
     object_file.write("l_vertical     =  True            # vertical line trigger                              \n")
     object_file.write("##=====================================================================================\n")
     object_file.write("# kpathname = [\"Gamma\",\"X\",\"Y\",\"Z\"] # Deprecated # N.B. edit it!               \n")
     object_file.write("#--------------------------------------------------------------------------------------\n")
     object_file.write("#STYPLE SETUPS                                                                         \n")
     object_file.write("lw_ud     = [ 3.0  , 3.0  ]               # linewidth for u/d bands                    \n")
     object_file.write("col_ud    = [ '#275BA9' , '#EE342E' ]     # color(and type) of u/d bands               \n")
     object_file.write("alpha0_ud = [ 1.00 , 1.00 ]               # alpha channel for...                       \n")
     object_file.write("col_ef    = 'k-' ; col_ver    = 'k:'      #---------------- for Ef and vertical line   \n")
     object_file.write("lw_ef     = 2.0  ; lw_ver     = 2.0                                                    \n")
     object_file.write("alpha0_ef = 0.5  ; alpha0_ver = 0.5                                                    \n")
     object_file.write("#--------------------------------------------------------------------------------------\n")
     object_file.write("#FIGURE SETUPS                                                                         \n")
     object_file.write("figsize0  = (12,14)                       # size of figure                             \n")
     object_file.write("# dpi0      = 60       # Deprecated         # dots per inch                            \n")
     object_file.write("fsize     = 20                            # font size                                  \n")
     object_file.write("##=====================================================================================\n")

def finder(keyword, file_):
    temp = []
    for line in file_:
        if keyword in line:
            print(line)
            temp.append(line[:-1])
    return temp

def kpath_parser(kpath_list):
    n_parts = len(kpath_list)
    kpath_temp = [part.split('-') for part in kpath_list]
    for idx, part in enumerate(kpath_temp):
        if idx+1 < n_parts:
            part[-1] = '{}/{}'.format(part[-1], kpath_temp[idx+1][0])
            kpath_temp[idx] = part
            kpath_temp[idx+1] = kpath_temp[idx+1][1:]
    
    kpath_ = []
    for part in kpath_temp:
        kpath_ += part

    for i in range(len(kpath_)):
        if "G" in kpath_[i]:
            kpath_[i] = kpath_[i].replace("G", "$\Gamma$")
        elif "g" in kpath_[i]:
            kpath_[i] = kpath_[i].replace("g", "$\Gamma$")
            
    return kpath_


# based on https://github.com/QijingZheng/pyband
def get_bandInfo_from_OUTCAR(OUTCAR_NAME="OUTCAR"):

    OUTCAR = [line for line in open(OUTCAR_NAME, "r") if line.strip()]
    
    for ln, line in enumerate(OUTCAR):
        if 'NKPTS =' in line:
            n_kpts = int(line.split()[3])
            n_bands = int(line.split()[-1])

        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])

        if "k-points in reciprocal lattice and weights" in line:
            start_idx_of_kpts_in_reciprocal_lattice = ln + 1

        if 'reciprocal lattice vectors' in line:
            start_idx_of_basis = ln + 1

        if 'E-fermi' in line:
            E_fermi = float(line.split()[2])
            start_idx_of_Efermi = ln + 1
    
    print("n_kpts  : {:d}".format(n_kpts))
    print("n_bands : {:d}".format(n_bands))
    print("ISPIN   : {:d}".format(ispin))
    print("E-fermi : {:.4f}".format(E_fermi))


    kpath_strings = OUTCAR[start_idx_of_kpts_in_reciprocal_lattice-1].split()[9:]
    kpath_info = kpath_parser(kpath_strings)
    

    B = np.array([line.split()[-3:] for line in OUTCAR[start_idx_of_basis:start_idx_of_basis+3]],
                 dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in OUTCAR[start_idx_of_kpts_in_reciprocal_lattice:start_idx_of_kpts_in_reciprocal_lattice+n_kpts]],
                   dtype=float)
    vec_kpts = tmp[:, :3]
    weight_kpts = tmp[:, -1]



    # number of band value lines
    N = (n_bands + 2) * n_kpts * ispin + (ispin - 1) * 2

    if 'Fermi energy:' in OUTCAR[start_idx_of_Efermi]:
        N += ispin

    bands = []
    for line in OUTCAR[start_idx_of_Efermi:start_idx_of_Efermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'Fermi energy:' in line:
            continue
        if 'k-point' in line:
            continue
        bands.append(float(line.split()[1]))

    bands = np.array(bands, dtype=float).reshape((ispin, n_kpts, n_bands))


    if os.path.isfile('KPOINTS'):
        kpoint_ = open('KPOINTS').readlines()

        if "line" in kpoint_[2]:
            n_points_per_line = int(kpoint_[1].split()[0])
            n_electrons = n_kpts // n_points_per_line
            vec_kpt_diff = np.zeros_like(vec_kpts, dtype=float)
    
            for i in range(n_electrons):
                start = i * n_points_per_line
                end = (i + 1) * n_points_per_line
                vec_kpt_diff[start:end, :] = vec_kpts[start:end, :] - vec_kpts[start, :]
    
            kpt_path = np.linalg.norm(np.dot(vec_kpt_diff, B), axis=1)
            for i in range(1, n_electrons):
                start = i * n_points_per_line
                end = (i + 1) * n_points_per_line
                kpt_path[start:end] += kpt_path[start-1]
    
            # kpt_path /= kpt_path[-1]
            kpt_bounds = np.concatenate((kpt_path[0::n_points_per_line], [kpt_path[-1], ]))

            if len(kpt_bounds) == len(kpath_info):
                print("K-Point : ",kpath_info)

            else:
                raise Exception("Number of K-Points are not matched")
            
        else:
            # get band path
            vec_kpt_diff = np.diff(vec_kpts, axis=0)
            kpt_path = np.zeros(n_kpts, dtype=float)
            kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vec_kpt_diff, B), axis=1))

            # get boundaries of band path
            xx = np.diff(kpt_path)
            kpt_bounds = np.concatenate(
            ([0.0, ], kpt_path[1:][np.isclose(xx, 0.0)], [kpt_path[-1], ]))


        return kpt_path, bands, E_fermi, kpt_bounds, weight_kpts, kpath_info, n_kpts, n_bands, ispin

    else:
        raise Exception("Can't find KPOINTS")
    
#-----------FUNCTIONS-----------------------------------------------------------------------------------------


if __name__=="__main__":
    #======================================================
    print( "##################################################################")
    print( "####  PPPV                                                   #####")
    print( "####  Primitive Program which Plots band for Vasp            #####")
    print( "#####   1/8/2012  Kyoo Kim & Prof. CJ Kang.                  #####")
    print( "#####   1/9/2023  Updated to latest Python by Donggeon Lee   #####")
    print( "##################################################################")

#-----------INPUT------------------------------------------------------
    input_name="BANDINP"
    iarg=1
    test_argv = [""]
    while iarg<len(test_argv):
        arg=sys.argv[iarg]
        if   arg=='-init' :
             if (is_valid_file(input_name)) :
                shutil.move(input_name,input_name+"_backup")
             object_file= open(input_name, 'w')
             input_template(object_file)
             print( "  >>> input template ",input_name," has been generated.")
             quit()
        else:     
             print( "" )
             print( ">> PPPV -init       : creats input file template.")
             quit()
        iarg+=1

    if (not is_valid_file(input_name)) :
        print( "  >>> [ERR] input file dose not exist.    ")
        print( "  >>>       We prepared a templit for ya. ")
        print( "  >>>       Please check and edit BANDINP.")
        inpfile = open(input_name,'w')
        input_template(inpfile) 
        quit()
    else:   
        exec(open(input_name).read())
    
    
    #======================================================

    kpt_path, bands, E_fermi, kpt_bounds, weight_kpts, kpath_info, n_kpts, n_bands, ispin = get_bandInfo_from_OUTCAR()
    bands -= E_fermi

    fig = plt.figure(figsize=figsize0)

    n_kpts_new = n_kpts * nkpt_intfac

    fat = dict(alpha=alpha0_ef)

    if nkpt_intfac > 1:
        d_kpts_new = np.double(max(kpt_path) - min(kpt_path)) / np.double(n_kpts_new - (len(kpt_bounds) - 1))
        klist_new = np.arange(np.double(min(kpt_path)), np.double(max(kpt_path)) + d_kpts_new, d_kpts_new)  # new mesh for interpolation
        
        split_position = [np.where(kpt_path==bound)[0] for bound in kpt_bounds]
        #new_kpt_bounds = [0]
        #for i in range(1, len(s1)-1):
        #    klist_new = np.insert(klist_new, split_position[i][1]*2-1, klist_new[split_position[i][1]*2-1])
        #    new_kpt_bounds.append(klist_new[split_position[i][1]*2-1])

        #new_kpt_bounds.append(klist_new[-1])
        
        for i_sp in range(ispin):  # SPIN LOOP
            print("  >>> spin", i_sp)
            for i_ban in range(n_bands):  # BAND LOOP
                fat = dict(alpha=alpha0_ud[i_sp])
                tck = interpolate.splrep(kpt_path, bands[i_sp].T[i_ban], s=0.05)  # get linear mesh from irregular one
                ene_new = interpolate.splev(klist_new, tck, der=0)  # get interpolated band on new mesh
                plt.plot(klist_new, ene_new, col_ud[i_sp], linewidth=lw_ud[i_sp], **fat)  # bandwise plot
            
        plt.plot(klist_new, np.zeros(np.shape(klist_new)), col_ef, linewidth=lw_ef, **fat)  # Ef
        
    else:
        for i_sp in range(ispin):
            for i in range(n_bands):
                plt.plot(kpt_path, bands.T[i], col_ud[i_sp], linewidth=lw_ud[i_sp], **fat)  # bandwise plot
        plt.plot(kpt_path, np.zeros(np.shape(kpt_path)), col_ef, linewidth=lw_ef, **fat)  # Ef

    # tics, vertical lines, ef....
    if l_vertical:
        fat = dict(alpha=alpha0_ver)
        #if nkpt_intfac > 1:
        #    vertial_x_list = new_kpt_bounds
        #else:
        #    vertial_x_list = kpt_bounds
        vertial_x_list = kpt_bounds
        for vertical_x in vertial_x_list[1:-1]:
            plt.plot((vertical_x, vertical_x), (e_min, e_max), col_ver, linewidth=lw_ver, **fat)

    # K-PATH-NAMES exception ..... expand for other case....

    yy_tic = list(np.arange(e_min, e_max + de, de))

    plt.xticks(vertial_x_list, kpath_info, color="k", size=fsize * 4.0 / 3.0)
    plt.yticks(yy_tic, color="k", size=fsize)

    if nkpt_intfac > 1:
        plt.xlim((min(klist_new), max(klist_new)))
    else:
        plt.xlim((min(kpt_path), max(kpt_path)))
    plt.ylim((e_min, e_max))
    plt.ylabel(r"$E - E_f$ [eV]", fontsize=fsize * 1.2)  # check figure offset! This does not appear in right position

    plt.savefig("band_structure.png", dpi=1000)
    plt.savefig("band_structure.eps", dpi=1000)
    plt.savefig("band_structure.svg", dpi=1000)