bl_info = {
    "name": "latk_ml_005", 
    "author": "Nick Fox-Gieg",
    "version": (0, 0, 1),
    "blender": (3, 0, 0),
    "description": "Generate brushstrokes from a mesh using informative-drawings, pix2pix, and vox2vox",
    "category": "Animation"
}

import bpy
import gpu
#import bgl
from bpy.types import Operator, AddonPreferences
from bpy.props import (BoolProperty, FloatProperty, StringProperty, IntProperty, PointerProperty, EnumProperty)
from bpy_extras.io_utils import (ImportHelper, ExportHelper)
import addon_utils
from mathutils import Vector, Matrix
import bmesh

import os
import sys
import subprocess
import platform
import argparse
import numpy as np
import latk_blender as lb

def runCmd(cmd, shell=False):
    returns = ""
    try:
        returns = subprocess.check_output(cmd, text=True, shell=shell)
    except subprocess.CalledProcessError as e:
        returns = f"Command failed with return code {e.returncode}"
    print(returns)
    return returns  

def getPythonExe():
    returns = None
    whichPlatform = platform.system().lower()
    
    if (whichPlatform == "darwin"):
        returns = os.path.join(sys.prefix, "bin", "python3.10")
    elif (whichPlatform == "windows"):
        returns = os.path.join(sys.prefix, "bin", "python.exe")
    else:
        returns = os.path.join(sys.prefix, "bin", "python3.10")
    
    return returns

def findAddonPath(name=None):
    #if not name:
        #name = __name__
    for mod in addon_utils.modules():
        if mod.bl_info["name"] == name:
            url = mod.__file__
            return os.path.dirname(url)
    return None


class latkml005Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    Backend: EnumProperty(
        name="Backend",
        items=(
            ("PYTORCH", "PyTorch", "...", 0),
            ("ONNX", "ONNX", "...", 1)
        ),
        default="PYTORCH"
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        row = box.row()
        row.operator("latkml005_button.install_requirements")

        box = layout.box()
        row = box.row()
        row.prop(self, "Backend")
        row.operator("latkml005_button.install_pytorch")
        row.operator("latkml005_button.install_onnx_cpu")
        row.operator("latkml005_button.install_onnx_gpu")

# This is needed to display the preferences menu
# https://docs.blender.org/api/current/bpy.types.AddonPreferences.html
class OBJECT_OT_latkml005_prefs(Operator):
    """Display example preferences"""
    bl_idname = "object.latkml005"
    bl_label = "latkml005 Preferences"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        return {'FINISHED'}


class latkml005Properties(bpy.types.PropertyGroup):
    """Properties for latkml005"""
    bl_idname = "GREASE_PENCIL_PT_latkml005Properties"

    SourceImage: EnumProperty(
        name="Source Image",
        items=(
            ("RGB", "RGB", "...", 0),
            ("Depth", "Depth", "...", 1)
        ),
        default="RGB"
    )

    ModelStyle1: EnumProperty(
        name="Model1",
        items=(
            ("ANIME", "Anime", "...", 0),
            ("CONTOUR", "Contour", "...", 1),
            ("OPENSKETCH", "OpenSketch", "...", 2),
            ("PXP_001", "PxP 001", "...", 3),
            ("PXP_002", "PxP 002", "...", 4),
            ("PXP_003", "PxP 003", "...", 5),
            ("PXP_004", "PxP 004", "...", 6)
        ),
        default="ANIME"
    )

    ModelStyle2: EnumProperty(
        name="Model2",
        items=(
            ("NONE", "None", "...", 0),
            ("ANIME", "Anime", "...", 1),
            ("CONTOUR", "Contour", "...", 2),
            ("OPENSKETCH", "OpenSketch", "...", 3)
        ),
        default="NONE"
    )    

    lineThreshold: FloatProperty(
        name="Line Threshold",
        description="...",
        default=32.0 #64.0
    )

    csize: IntProperty(
        name="csize",
        description="...",
        default=10
    )

    maxIter: IntProperty(
        name="iter",
        description="...",
        default=999
    )

    distThreshold: FloatProperty(
        name="Dist Threshold",
        description="...",
        default=0.1 #0.5
    )

    thickness: FloatProperty(
        name="Thickness %",
        description="...",
        default=10.0
    )

    Operation1: EnumProperty(
        name="Operation 1",
        items=(
            ("NONE", "None", "...", 0),
            ("64_VOXEL", "64^3 voxels", "...", 1),
            ("128_VOXEL", "128^3 voxels", "...", 2),
            ("256_VOXEL", "256^3 voxels", "...", 3)
        ),
        default="NONE"
    )

    Operation2: EnumProperty(
        name="Operation 2",
        items=(
            ("NONE", "None", "...", 0),
            ("GET_EDGES", "Get Edges", "...", 1)
        ),
        default="NONE"
    )

    Operation3: EnumProperty(
        name="Operation 3",
        items=(
            ("STROKE_GEN", "Connect Strokes", "...", 0),
            ("CONTOUR_GEN", "Connect Contours", "...", 1),
            ("SKEL_GEN", "Connect Skeleton", "...", 2)
        ),
        default="STROKE_GEN"
    )

    do_filter: BoolProperty(
        name="Prefilter",
        description="...",
        default=True
    )

    do_modifiers: BoolProperty(
        name="Modifiers",
        description="...",
        default=True
    )

    do_recenter: BoolProperty(
        name="Recenter",
        description="...",
        default=False
    )

    dims: IntProperty(
        name="Dims",
        description="Voxel Dimensions",
        default=256
    )

    strokegen_radius: FloatProperty(
        name="StrokeGen Radius",
        description="Base search distance for points",
        default=0.05
    )

    strokegen_minPointsCount: IntProperty(
        name="StrokeGen Min Points",
        description="Minimum number of points to make a stroke",
        default=5
    )


class latkml005_Button_InstallRequirements(bpy.types.Operator):
    bl_idname = "latkml005_button.install_requirements"
    bl_label = "Install Requirements"
    
    def execute(self, context):
        python_exe = getPythonExe()        
        whichPlatform = platform.system().lower()
        root_url = findAddonPath(__name__)
        runCmd([python_exe, "-m", "pip", "install", "-r", os.path.join(root_url, "requirements.txt")])

        if (whichPlatform == "darwin"):
            runCmd(["bash", os.path.join(root_url, "skeleton_tracing/swig/compile.command")])
        elif (whichPlatform == "windows"):
            runCmd([os.path.join(root_url, "skeleton_tracing/swig/compile.bat")])
        else:
            runCmd(["bash", os.path.join(root_url, "skeleton_tracing/swig/compile.sh")])

        return {'FINISHED'}


class latkml005_Button_InstallOnnxCpu(bpy.types.Operator):
    bl_idname = "latkml005_button.install_onnx_cpu"
    bl_label = "Install ONNX CPU"
    
    def execute(self, context):
        python_exe = getPythonExe()
        runCmd([python_exe, "-m", "pip", "uninstall", "onnxruntime-gpu"])
        runCmd([python_exe, "-m", "pip", "install", "onnxruntime"])
        return {'FINISHED'}


class latkml005_Button_InstallOnnxGpu(bpy.types.Operator):
    bl_idname = "latkml005_button.install_onnx_gpu"
    bl_label = "Install ONNX GPU"
    
    def execute(self, context):
        python_exe = getPythonExe()
        runCmd([python_exe, "-m", "pip", "uninstall", "onnxruntime"])
        runCmd([python_exe, "-m", "pip", "install", "onnxruntime-gpu"])
        return {'FINISHED'}


class latkml005_Button_InstallPytorch(bpy.types.Operator):
    bl_idname = "latkml005_button.install_pytorch"
    bl_label = "Install Pytorch"
    
    def execute(self, context):       
        python_exe = getPythonExe()
        whichPlatform = platform.system().lower()
        
        if (whichPlatform == "darwin"):
            runCmd([python_exe, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/cpu'])
        else:
            runCmd([python_exe, '-m', 'pip', 'install', '--upgrade', 'torch', 'torchvision', 'torchaudio', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])

        return {'FINISHED'}


class latkml005_Button_AllFrames_003(bpy.types.Operator):
    from . import latk_ml
    """Operate on all frames"""
    bl_idname = "latkml005_button.allframes003"
    bl_label = "003 All"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latk_ml.doVoxelOpCore(__name__, context, allFrames=True)
        return {'FINISHED'}


class latkml005_Button_SingleFrame_003(bpy.types.Operator):
    from . import latk_ml
    """Operate on a single frame"""
    bl_idname = "latkml005_button.singleframe003"
    bl_label = "003 Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latk_ml.doVoxelOpCore(__name__, context, allFrames=False)
        return {'FINISHED'}


class latkml005_Button_AllFrames_004(bpy.types.Operator):
    from . import latk_ml
    """Operate on all frames"""
    bl_idname = "latkml005_button.allframes004"
    bl_label = "004 All"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = latk_ml.loadModel004(__name__)

        la = latk_ml.latk.Latk()
        la.layers.append(latk_ml.latk.LatkLayer())

        start, end = lb.getStartEnd()
        for i in range(start, end):
            lb.goToFrame(i)
            laFrame = latk_ml.doInference004(net1, net2)
            la.layers[0].frames.append(laFrame)

        lb.fromLatkToGp(la, resizeTimeline=False)
        lb.setThickness(latkml005.thickness)
        return {'FINISHED'}


class latkml005_Button_SingleFrame_004(bpy.types.Operator):
    from . import latk_ml
    """Operate on a single frame"""
    bl_idname = "latkml005_button.singleframe004"
    bl_label = "004 Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = latk_ml.loadModel004(__name__)

        la = latk_ml.latk.Latk()
        la.layers.append(latk_ml.latk.LatkLayer())
        laFrame = latk_ml.doInference004(net1, net2)
        la.layers[0].frames.append(laFrame)
        
        lb.fromLatkToGp(la, resizeTimeline=False)
        lb.setThickness(latkml005.thickness)
        return {'FINISHED'}


# https://blender.stackexchange.com/questions/167862/how-to-create-a-button-on-the-n-panel
class latkml005Properties_Panel(bpy.types.Panel):
    """Creates a Panel in the 3D View context"""
    bl_idname = "GREASE_PENCIL_PT_latkml005PropertiesPanel"
    bl_space_type = 'VIEW_3D'
    bl_label = "latk_ml_005"
    bl_category = "Latk"
    bl_region_type = 'UI'
    #bl_context = "objectmode" # "mesh_edit"

    #def draw_header(self, context):
        #self.layout.prop(context.scene.freestyle_gpencil_export, "enable_latk", text="")

    def draw(self, context):
        latkml005 = context.scene.latkml005_settings

        layout = self.layout

        box = layout.box()

        row = box.row()
        row.operator("latkml005_button.singleframe004")
        row.operator("latkml005_button.allframes004")

        row = box.row()
        row.prop(latkml005, "ModelStyle1")

        row = box.row()
        row.prop(latkml005, "ModelStyle2")

        row = box.row()
        row.prop(latkml005, "lineThreshold")

        row = box.row()
        row.prop(latkml005, "distThreshold")

        row = box.row()
        row.prop(latkml005, "csize")
        row.prop(latkml005, "maxIter")

        row = box.row()
        row.prop(latkml005, "thickness")

        row = box.row()
        row.prop(latkml005, "SourceImage")

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        box = layout.box()

        row = box.row()
        row.operator("latkml005_button.singleframe003")
        row.operator("latkml005_button.allframes003")

        if (bpy.context.preferences.addons[__name__].preferences.Backend.lower() == "pytorch"):
            row = box.row()
            row.prop(latkml005, "Operation1")

            row = box.row()
            row.prop(latkml005, "do_filter")
            row.prop(latkml005, "do_modifiers")
            row.prop(latkml005, "do_recenter")

        row = box.row()
        row.prop(latkml005, "Operation2")

        row = box.row()
        row.prop(latkml005, "Operation3")
        row = box.row()
        row.prop(latkml005, "thickness")
        row = box.row()
        row.prop(latkml005, "strokegen_radius")
        row.prop(latkml005, "strokegen_minPointsCount")

classes = (
    OBJECT_OT_latkml005_prefs,
    latkml005Preferences,
    latkml005Properties,
    latkml005Properties_Panel,
    latkml005_Button_AllFrames_004,
    latkml005_Button_SingleFrame_004,
    latkml005_Button_AllFrames_003,
    latkml005_Button_SingleFrame_003,
    latkml005_Button_InstallRequirements,
    latkml005_Button_InstallOnnxCpu,
    latkml005_Button_InstallOnnxGpu,
    latkml005_Button_InstallPytorch
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)   
    bpy.types.Scene.latkml005_settings = bpy.props.PointerProperty(type=latkml005Properties)

def unregister():
    del bpy.types.Scene.latkml005_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()