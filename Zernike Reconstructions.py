"""
Zernike Pipeline for MS39 Data Processing v2.5 - Industry Standard Centering
-----------------------------------------------------------------------------
Version corrigée pour implémenter une analyse Zernike standard, centrée sur 
le vertex de la sphère de meilleure approximation (Best Fit Sphere - BFS).
Cette méthode est conçue pour répliquer les résultats standards de l'industrie
et fournir des coefficients d'aberration physiologiquement réalistes.
"""

import os
import pandas as pd
import numpy as np
import math
from scipy.optimize import least_squares
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================================
# PART 1: DATA PROCESSING AND ZERNIKE ANALYSIS
# ============================================================================
segments = [('elevation_anterior',316,27,-999,10000),('elevation_posterior',348,26,-999,10000),('elevation_stromal',380,25,-999,10000)]
def lire_segment(f,d,n):
    try: return pd.read_csv(f,sep=';',header=None,skiprows=d,nrows=n,usecols=range(256),dtype=float,encoding='latin1')
    except Exception as e: print(f"[ERREUR] Lecture segment ({d}-{d+n}): {e}"); return pd.DataFrame()

def extract_diameter_zones(sm,d):
    r,nr=d/2.0,int(d/2.0/0.2)+1
    if nr>len(sm):return None
    return sm.iloc[:nr].values

def process_single_csv_zones(p):
    r={'filename':os.path.basename(p),'surfaces':{}}
    for sn,sr,nr,mn,mx in segments:
        ss,m=sn.replace('elevation_',''),lire_segment(p,sr,nr)
        if m.empty:continue
        m=m.mask((m<mn)|(m>mx)).replace(-1000,np.nan)
        r['surfaces'][ss]={}
        for dia in [2,3,4,5,6,7]:
            zd=extract_diameter_zones(m,dia)
            if zd is not None:r['surfaces'][ss][f'{dia}mm']={'data':zd}
    return r

def polar_to_cartesian_coords(zd,rs=0.2):
    nr,na=zd.shape
    rad,ang=np.arange(nr)*rs,np.linspace(0,2*np.pi,na,endpoint=False)
    rg,tg=np.meshgrid(rad,ang,indexing='ij')
    vm=~np.isnan(zd)
    rf,tf,zf=rg[vm],tg[vm],zd[vm]
    return rf*np.cos(tf),rf*np.sin(tf),zf

def sphere_function(p,x,y):
    cx,cy,cz,R=p
    d=R**2-(x-cx)**2-(y-cy)**2
    # Ensure argument to sqrt is non-negative
    return cz-np.sqrt(np.maximum(d,0))

def bfs_residuals(p,x,y,z):return z-sphere_function(p,x,y)

def calculate_bfs(zd,rs=0.2):
    x,y,z=polar_to_cartesian_coords(zd,rs)
    if len(x)<4:return None,None
    zr=np.max(z)-np.min(z)
    er=np.clip((np.max(np.sqrt(x**2+y**2))**2)/(2*zr),6.,15.) if zr>1e-3 else 8.
    ig=[0.,0.,np.max(z)+er,er]
    try:
        res=least_squares(bfs_residuals,ig,args=(x,y,z),method='lm',max_nfev=1000)
        if not res.success:return None,None
        p=res.x
        if not (abs(p[0])<5 and abs(p[1])<5 and 4.<p[3]<20.):return None,None
        bp={'center_x':p[0],'center_y':p[1],'center_z':p[2],'radius':p[3],'rms_residual':np.sqrt(np.mean(res.fun**2))}
        fi={'n_points':len(x),'convergence':res.success}
        return bp,fi
    except Exception:return None,None

def calculate_residual_map(zd,bp,rs=0.2):
    nr,na=zd.shape
    rm=np.full_like(zd,np.nan)
    rad,ang=(np.arange(nr)*rs)[:,np.newaxis],np.linspace(0,2*np.pi,na,endpoint=False)[np.newaxis,:]
    x,y=rad*np.cos(ang),rad*np.sin(ang)
    zs=sphere_function(bp,x,y)
    vm=~np.isnan(zd)
    rm[vm]=zd[vm]-zs[vm]
    return rm

def zernike_radial(n,m,rho):
    if(n-abs(m))%2!=0:return np.zeros_like(rho)
    R=np.zeros_like(rho)
    for k in range((n-abs(m))//2+1):
        c=((-1)**k*math.factorial(n-k)/(math.factorial(k)*math.factorial((n+abs(m))//2-k)*math.factorial((n-abs(m))//2-k)))
        R+=c*(rho**(n-2*k))
    return R

def zernike_polynomial(n, m, rho, theta):
    R = zernike_radial(n, abs(m), rho)
    if m == 0:
        norm_factor = math.sqrt(n + 1)
    else:
        norm_factor = math.sqrt(2 * (n + 1))
    if m >= 0:
        return norm_factor * R * np.cos(m * theta)
    else:
        return norm_factor * R * np.sin(abs(m) * theta)

def generate_zernike_modes(mo=6):
    m,j,zn=[],0,{(0,0):'piston',(1,-1):'tilt_y',(1,1):'tilt_x',(2,-2):'astigmatism_oblique',(2,0):'defocus',(2,2):'astigmatism_vertical',(3,-3):'trefoil_y',(3,-1):'coma_y',(3,1):'coma_x',(3,3):'trefoil_x',(4,-4):'tetrafoil_y',(4,-2):'secondary_astigmatism_oblique',(4,0):'spherical_aberration',(4,2):'secondary_astigmatism_vertical',(4,4):'tetrafoil_x',(5,-5):'pentafoil_y',(5,-3):'secondary_trefoil_y',(5,-1):'secondary_coma_y',(5,1):'secondary_coma_x',(5,3):'secondary_trefoil_x',(5,5):'pentafoil_x',(6,-6):'hexafoil_y',(6,-4):'secondary_tetrafoil_y',(6,-2):'tertiary_astigmatism_oblique',(6,0):'secondary_spherical_aberration',(6,2):'tertiary_astigmatism_vertical',(6,4):'secondary_tetrafoil_x',(6,6):'hexafoil_x'}
    for n in range(mo+1):
        for o in range(-n,n+1,2):m.append((j,n,o,zn.get((n,o),f'Z{n}_{o}')));j+=1
    return m

def fit_zernike_coefficients(x,y,z,mo,pr):
    vm=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    xv,yv,zv=x[vm],y[vm],z[vm]
    if len(xv)<10:return None,None,None
    # rho and theta are now calculated from the centered (x,y) coordinates
    rho,theta=np.sqrt(xv**2+yv**2)/pr,np.arctan2(yv,xv)
    i=rho<=1.001
    rho,theta,zv=rho[i],theta[i],zv[i]
    if len(rho)<10:return None,None,None
    modes=generate_zernike_modes(mo)
    A=np.vstack([zernike_polynomial(n,m,rho,theta) for j,n,m,name in modes]).T
    try:
        c,_,_,_=np.linalg.lstsq(A,zv,rcond=None)
        if not np.all(np.isfinite(c)):return None,None,None
        zf=A@c
        ve=100*(1-np.sum((zv-zf)**2)/np.sum((zv-np.mean(zv))**2)) if np.var(zv)>1e-12 else 100.
        fi={'rms_residual':np.sqrt(np.mean((zv-zf)**2)),'n_points_fitted':len(rho),'variance_explained_pct':ve}
        return c,modes,fi
    except np.linalg.LinAlgError:return None,None,None

# ============================================================================
# PART 1.5: OPD CORRECTION FUNCTION
# ============================================================================
def apply_opd_and_convention_correction(coefficients_mm, modes):
    # This factor converts geometric elevation (in mm) to optical path difference (in mm)
    # n_cornea = 1.376, n_air = 1.000. Factor is (n_air - n_cornea) = -0.376
    correction_factor = -0.376 
    corrected_coefficients_mm = np.array(coefficients_mm) * correction_factor
    
    # Per convention, the piston term in OPD is set to zero
    if len(corrected_coefficients_mm) > 0:
        corrected_coefficients_mm[0] = 0.0
    return corrected_coefficients_mm

def save_results_to_csv(ar, of):
    print(f"\n{'='*80}\nSAVING RESULTS TO CSV\n{'='*80}")
    sf = {k: v for k, v in ar['files'].items() if v.get('status') == 'success'}
    if not sf:
        print("[WARNING] No successful files were processed.")
        return

    all_geom_results, all_opd_results = [], []
    rm = generate_zernike_modes()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for fn, fd in sf.items():
        geom_file_row = {'filename': fn}
        opd_file_row = {'filename': fn}
        
        for sn, sd in fd.get('zernike_results', {}).get('surfaces', {}).items():
            for zn, zd in sd.items():
                p = f"{sn}_{zn}"
                b = zd.get('bfs_params', {})
                bfi = zd.get('bfs_fit_info', {})
                zi = zd.get('zernike_fit_info', {})
                c_geom = zd.get('coefficients_geometric_mm')
                c_opd = zd.get('coefficients_opd_corrected_mm')

                # Common info for both file types
                common_info = {
                    f'{p}_bfs_convergence': bfi.get('convergence'),
                    f'{p}_bfs_fitted_points': bfi.get('n_points'),
                    f'{p}_bfs_radius_mm': b.get('radius'),
                    f'{p}_bfs_center_x_mm': b.get('center_x'),
                    f'{p}_bfs_center_y_mm': b.get('center_y'),
                    f'{p}_bfs_center_z_mm': b.get('center_z'),
                    f'{p}_bfs_rms_residual_mm': b.get('rms_residual'),
                    f'{p}_zernike_fitted_points': zi.get('n_points_fitted'),
                    f'{p}_zernike_rms_residual_um': zi.get('rms_residual', np.nan) * 1000,
                    f'{p}_zernike_variance_explained_pct': zi.get('variance_explained_pct')
                }
                geom_file_row.update(common_info)
                opd_file_row.update(common_info)

                # Add specific Zernike columns for each file type
                for j, n, m, name in rm:
                    geom_file_row[f'{p}_zernike_z{j}_{name}_um'] = c_geom[j] * 1000 if c_geom is not None and j < len(c_geom) else np.nan
                    opd_file_row[f'{p}_zernike_z{j}_{name}_um'] = c_opd[j] * 1000 if c_opd is not None and j < len(c_opd) else np.nan

        all_geom_results.append(geom_file_row)
        all_opd_results.append(opd_file_row)

    # Save Geometric Results CSV
    if all_geom_results:
        df_geom = pd.DataFrame(all_geom_results)
        ofp_geom = os.path.join(of, f"batch_results_geom_{ts}.csv")
        df_geom.to_csv(ofp_geom, index=False, sep=';', float_format='%.6f')
        print(f"\n[SUCCESS] Geometric results saved to:\n  {ofp_geom}")
    
    # Save OPD Results CSV
    if all_opd_results:
        df_opd = pd.DataFrame(all_opd_results)
        ofp_opd = os.path.join(of, f"batch_results_opd_{ts}.csv")
        df_opd.to_csv(ofp_opd, index=False, sep=';', float_format='%.6f')
        print(f"[SUCCESS] OPD results saved to:\n  {ofp_opd}")


# ============================================================================
# PART 2: VISUALIZATION FUNCTIONS
# ============================================================================
colors=['#57597e','#71466B','#8A3259','#a41f46','#B60F37','#c80029','#D80017','#e90005','#EC1B02','#ef3500','#F05400','#f27300','#ED8D00','#e9a700','#EFB500','#f6c300','#F9CE00','#fbd900','#A0D900','#46d900','#4ADE7E','#4ee2fd','#4BD3FE','#48c4ff','#45B9FF','#41aeff','#3CA4FE','#389afc','#338EF2','#2e81e8','#2A73E9','#2664e9','#254DF4','#2435ff','#2C22F2','#340ee5','#4C07D0','#6300bb','#8400BA','#a400ba']
custom_cmap=mcolors.ListedColormap(colors)
custom_cmap.set_bad('white',1.0)

def recreate_zernike_map(coefficient_um,n,m,diameter_mm,grid_size=256):
    radius_mm=diameter_mm/2.0
    x,y=np.linspace(-radius_mm,radius_mm,grid_size),np.linspace(-radius_mm,radius_mm,grid_size)
    X,Y=np.meshgrid(x,y)
    rho,theta=np.sqrt(X**2+Y**2),np.arctan2(Y,X)
    rho_norm=rho/radius_mm
    zernike_map=coefficient_um*zernike_polynomial(n,m,rho_norm,theta)
    zernike_map[rho_norm>1.001]=np.nan
    return zernike_map

def plot_dynamic_map(map_data,output_filepath):
    vmin,vmax=np.nanmin(map_data),np.nanmax(map_data)
    if np.isclose(vmin,vmax):vmin-=0.1;vmax+=0.1
    levels=np.linspace(vmin,vmax,40)
    norm=mcolors.BoundaryNorm(levels,custom_cmap.N)
    fig,ax=plt.subplots(figsize=(5,5))
    ax.imshow(map_data,cmap=custom_cmap,norm=norm,origin='lower',interpolation='bicubic')
    ax.axis('off')
    fig.tight_layout(pad=0)
    plt.savefig(output_filepath,dpi=150)
    plt.close(fig)

def generate_visualizations(zernike_results,output_folder):
    print(f"\n{'='*80}\nGENERATING DYNAMIC VISUALIZATIONS\n{'='*80}")
    base_output_folder=os.path.join(output_folder,"Zernike_Maps_Dynamique")
    for surface,s_data in zernike_results['surfaces'].items():
        for diameter,z_data in s_data.items():
            dia_mm=float(diameter[:-2])
            coeffs_geom_mm, modes = z_data.get('coefficients_geometric_mm'), z_data.get('modes')
            if coeffs_geom_mm is None or modes is None:continue
            
            vis_output_folder=os.path.join(base_output_folder,f"{surface}_{diameter}")
            os.makedirs(vis_output_folder,exist_ok=True)
            print(f"  -> Creating dynamic maps for {surface} @ {diameter}...")
            
            for j,n,m,name in modes:
                # We usually visualize higher-order aberrations
                if j<=2:continue 
                map_data=recreate_zernike_map(coeffs_geom_mm[j]*1000,n,m,dia_mm)
                filename=f"z{j:02d}_{name}.png"
                plot_dynamic_map(map_data,os.path.join(vis_output_folder,filename))
    print(f"\n[SUCCESS] All dynamic visualizations saved in:\n  {base_output_folder}")

# ============================================================================
# PART 3: MAIN EXECUTION PIPELINE
# ============================================================================
# <<< MODIFIED AND CORRECTED FUNCTION >>>
def run_full_pipeline(input_dir,output_dir,max_order=6):
    os.makedirs(output_dir,exist_ok=True)
    print(f"[INFO] Reading raw data from: {input_dir}")
    print(f"[INFO] All results will be saved to: {output_dir}")
    all_results={'files':{}}
    csv_files=[f for f in os.listdir(input_dir) if f.endswith('.csv') and not f.startswith('batch_results_')]
    if not csv_files:print("[ERROR] No valid input CSV files found.");return
    
    for i,fname in enumerate(csv_files,1):
        print(f"\n--- Processing file {i}/{len(csv_files)}: {fname} ---")
        try:
            results={'filename':fname,'surfaces':{}}
            zone_results=process_single_csv_zones(os.path.join(input_dir,fname))
            
            for s_name,s_data in zone_results['surfaces'].items():
                results['surfaces'][s_name]={}
                for z_name,z_data in s_data.items():
                    # <<< MODIFICATION START: CORRECTED LOGIC >>>
                    
                    # STEP 1: Calculate Best Fit Sphere (BFS) on the original data.
                    # This determines the reference surface and its center (cx, cy).
                    bfs_params,bfs_info=calculate_bfs(z_data['data'])
                    if bfs_params is None:
                        print(f"  ... SKIPPED {s_name} @ {z_name}: BFS fitting failed.")
                        continue
                    
                    cx = bfs_params['center_x']
                    cy = bfs_params['center_y']
                    print(f"  ... BFS for {s_name} @ {z_name} centered at (x={cx:.4f}, y={cy:.4f}) mm")

                    # STEP 2: Calculate the residual map by subtracting the BFS.
                    # The Zernike fit will be performed on this residual data.
                    residual_map=calculate_residual_map(z_data['data'], [cx, bfs_params['center_y'], bfs_params['center_z'], bfs_params['radius']])
                    
                    # Get cartesian coordinates of the residual map.
                    x, y, z = polar_to_cartesian_coords(residual_map)
                    
                    # STEP 3: Recenter the coordinate system on the BFS Vertex (cx, cy).
                    # This is the crucial step for a physically meaningful analysis.
                    # We are analyzing the residual surface (z) in a coordinate system
                    # that is centered on the axis of the sphere we just removed.
                    x_centered = x - cx
                    y_centered = y - cy
                    
                    # STEP 4: Fit Zernike coefficients on the correctly centered data.
                    # The z values (residuals) and the (x, y) coordinates are now consistent.
                    coeffs_geom_mm, modes, zern_info = fit_zernike_coefficients(
                        x_centered, y_centered, z, max_order, float(z_name[:-2])/2.0
                    )
                    # <<< MODIFICATION END >>>
                    
                    if coeffs_geom_mm is not None and modes is not None:
                        corrected_opd_coeffs_mm = apply_opd_and_convention_correction(coeffs_geom_mm, modes)
                    else:
                        corrected_opd_coeffs_mm = None
                    
                    results['surfaces'][s_name][z_name]={
                        'coefficients_geometric_mm': coeffs_geom_mm,
                        'coefficients_opd_corrected_mm': corrected_opd_coeffs_mm,
                        'modes': modes,
                        'zernike_fit_info': zern_info,
                        'bfs_params': bfs_params,
                        'bfs_fit_info': bfs_info
                    }
            
            all_results['files'][fname]={'status':'success','zernike_results':results}
            print(f"✓ SUCCESS: {fname} processed.")
        except Exception as e:
            all_results['files'][fname]={'status':'failed','error':str(e)}
            print(f"✗ FAILED: {fname} - {e}")
            
    save_results_to_csv(all_results,output_dir)
    
    for fname,data in all_results['files'].items():
        if data.get('status')=='success':
            file_output_dir=os.path.join(output_dir,os.path.splitext(fname)[0])
            generate_visualizations(data['zernike_results'],file_output_dir)

if __name__=="__main__":
    print("[DEBUG] Starting Industry-Standard Zernike pipeline.")
    # Mettez à jour ces chemins d'accès selon votre configuration
    input_folder=r'C:\Users\nassd\OneDrive\Bureau\15-20\moi'
    output_folder=r'C:\Users\nassd\OneDrive\Bureau\15-20\Zernike_Pipeline_Output_Correctedd'
    run_full_pipeline(input_dir=input_folder,output_dir=output_folder,max_order=6)
    print(f"\n[INFO] Batch processing and visualization completed!")