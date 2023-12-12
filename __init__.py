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
import bgl
from bpy.types import Operator, AddonPreferences
from bpy.props import (BoolProperty, FloatProperty, StringProperty, IntProperty, PointerProperty, EnumProperty)
from bpy_extras.io_utils import (ImportHelper, ExportHelper)
import addon_utils

import os
import sys
import argparse
import cv2
import numpy as np
import latk
import latk_blender as lb
from skimage.morphology import skeletonize
from mathutils import Vector, Quaternion
from collections import namedtuple

import onnxruntime as ort
import torch

import bmesh
from mathutils import Vector, Matrix
import h5py

import random

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import itertools
from torchvision import datasets

import skeletor as sk
import trimesh
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import scipy.ndimage as nd
from pyntcloud import PyntCloud 
import pandas as pd
import pdb

def findAddonPath(name=None):
    if not name:
        name = __name__
    for mod in addon_utils.modules():
        if mod.bl_info["name"] == name:
            url = mod.__file__
            return os.path.dirname(url)
    return None

from .skeleton_tracing.swig.trace_skeleton import *
from .informative_drawings.model import Generator 
from .pix2pix.models import pix2pix_model

#from . import binvox_rw
from .vox2vox import binvox_rw
from .vox2vox.models import *
from .vox2vox.dataset import CTDataset


class latkml005Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    '''
    extraFormats_AfterEffects: bpy.props.BoolProperty(
        name = 'After Effects JSX',
        description = "After Effects JSX export",
        default = False
    )
    '''

    def draw(self, context):
        layout = self.layout

        layout.label(text="none")
        #row = layout.row()
        #row.prop(self, "extraFormats_Painter")


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

    Backend: EnumProperty(
        name="Backend",
        items=(
            ("ONNX", "ONNX", "...", 0),
            ("PYTORCH", "PyTorch", "...", 1)
        ),
        default="PYTORCH"
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
    '''
    lineThreshold = 64
    csize = 10
    maxIter = 999
    '''

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


class latkml005_Button_AllFrames_003(bpy.types.Operator):
    """Operate on all frames"""
    bl_idname = "latkml005_button.allframes003"
    bl_label = "003 All"
    bl_options = {'UNDO'}
    
    def execute(self, context):

        doVoxelOpCore(context, allFrames=True)

        return {'FINISHED'}


class latkml005_Button_SingleFrame_003(bpy.types.Operator):
    """Operate on a single frame"""
    bl_idname = "latkml005_button.singleframe003"
    bl_label = "003 Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        doVoxelOpCore(context, allFrames=False)
        return {'FINISHED'}


class latkml005_Button_AllFrames_004(bpy.types.Operator):
    """Operate on all frames"""
    bl_idname = "latkml005_button.allframes004"
    bl_label = "004 All"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = loadModel004()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())

        start, end = lb.getStartEnd()
        for i in range(start, end):
            lb.goToFrame(i)
            laFrame = doInference004(net1, net2)
            la.layers[0].frames.append(laFrame)

        lb.fromLatkToGp(la, resizeTimeline=False)
        lb.setThickness(latkml005.thickness)
        return {'FINISHED'}


class latkml005_Button_SingleFrame_004(bpy.types.Operator):
    """Operate on a single frame"""
    bl_idname = "latkml005_button.singleframe004"
    bl_label = "004 Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = loadModel004()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())
        laFrame = doInference004(net1, net2)
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

        row = layout.row()
        row.operator("latkml005_button.singleframe004")
        row.operator("latkml005_button.allframes004")

        row = layout.row()
        row.prop(latkml005, "ModelStyle1")

        row = layout.row()
        row.prop(latkml005, "ModelStyle2")

        row = layout.row()
        row.prop(latkml005, "lineThreshold")

        row = layout.row()
        row.prop(latkml005, "distThreshold")

        row = layout.row()
        row.prop(latkml005, "csize")
        row.prop(latkml005, "maxIter")

        row = layout.row()
        row.prop(latkml005, "thickness")

        row = layout.row()
        row.prop(latkml005, "SourceImage")

        row = layout.row()
        row.prop(latkml005, "Backend")

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        row = layout.row()
        row.operator("latkml005_button.singleframe003")
        row.operator("latkml005_button.allframes003")

        row = layout.row()
        row.prop(latkml005, "Operation1")

        row = layout.row()
        row.prop(latkml005, "do_filter")
        row.prop(latkml005, "do_recenter")

        row = layout.row()
        row.prop(latkml005, "Operation2")

        row = layout.row()
        row.prop(latkml005, "Operation3")
        row = layout.row()
        row.prop(latkml005, "thickness")
        row = layout.row()
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
    latkml005_Button_SingleFrame_003
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

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def npToCv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cvToNp(img):
    return np.asarray(img)

def cvToBlender(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    blender_image = bpy.data.images.new("Image", width=rgb_image.shape[1], height=rgb_image.shape[0])
    pixels = np.flip(rgb_image.flatten())
    blender_image.pixels.foreach_set(pixels)
    blender_image.update()
    return blender_image

def createTempOutputSettings(newFilename="render.png", newFormat="PNG"):
    newFilepath = os.path.join(bpy.app.tempdir, newFilename)

    oldFilepath = bpy.context.scene.render.filepath
    oldFormat = bpy.context.scene.render.image_settings.file_format
    
    bpy.context.scene.render.filepath = newFilepath
    bpy.context.scene.render.image_settings.file_format = newFormat

    return oldFilepath, oldFormat, newFilepath, newFormat

def restoreOldOutputSettings(oldFilepath, oldFormat):
    bpy.context.scene.render.filepath = oldFilepath
    bpy.context.scene.render.image_settings.file_format = oldFormat

def renderFrame(depthPass=False):
    oldFilepath, oldFormat, newFilepath, newFormat = createTempOutputSettings()
    if (depthPass == True):
        setupDepthPass(newFilepath.split("." + newFormat)[0] + "_depth")
    bpy.ops.render.render(write_still=True)
    restoreOldOutputSettings(oldFilepath, oldFormat)
    if (depthPass == True):
        return os.path.join(newFilepath.split("." + newFormat)[0] + "_depth", "Image" + str(bpy.data.scenes['Scene'].frame_current).zfill(4) + "." + newFormat)
    else:
        return newFilepath

# https://www.saifkhichi.com/blog/blender-depth-map-surface-normals
def getDepthPassAlt():
    """Obtains depth map from Blender render.
    :return: The depth map of the rendered camera view as a numpy array of size (H,W).
    """
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
    dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    return dmap

# https://blender.stackexchange.com/questions/2170/how-to-access-render-result-pixels-from-python-script/23309#23309
# https://blender.stackexchange.com/questions/56967/how-to-get-depth-data-using-python-api
def setupDepthPass(url="/my_path/"):
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    #~
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    #~
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    #~
    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.size = [0.08]
    map.use_min = True
    map.min = [0]
    map.use_max = True
    map.max = [255]
    links.new(rl.outputs[2], map.inputs[0])
    #~
    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])
    #~
    # The viewer can come in handy for inspecting the results in the GUI
    depthViewer = tree.nodes.new(type="CompositorNodeViewer")
    links.new(invert.outputs[0], depthViewer.inputs[0])
    # Use alpha from input.
    links.new(rl.outputs[1], depthViewer.inputs[1])
    #~
    # create a file output node and set the path
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = url
    links.new(invert.outputs[0], fileOutput.inputs[0])

def renderToCv(depthPass=False):
    image_path = renderFrame(depthPass)
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def renderToNp(depthPass=False):
    image_path = renderFrame(depthPass)
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    image = bpy.data.images.load(image_path)
    image_array = np.array(image.pixels[:])
    image_array = image_array.reshape((height, width, 4))
    image_array = np.flipud(image_array)
    image_array = image_array[:, :, :3]
    return image_array.astype(np.float32)

def getModelPath(url):
    return os.path.join(findAddonPath(), url)

def loadModel003():
    latkml005 = bpy.context.scene.latkml005_settings
    returns = modelSelector003(latkml005.Operation1)
    return returns

def loadModel004():
    latkml005 = bpy.context.scene.latkml005_settings
   
    returns1 = modelSelector004(latkml005.ModelStyle1)
    returns2 = modelSelector004(latkml005.ModelStyle2)

    return returns1, returns2

def modelSelector003(modelName):
    latkml003 = bpy.context.scene.latkml005_settings

    modelName = modelName.lower()
    latkml003.dims = int(modelName.split("_")[0])
    return Vox2Vox_PyTorch("model/" + modelName + ".pth")

def modelSelector004(modelName):
    modelName = modelName.lower()
    latkml005 = bpy.context.scene.latkml005_settings

    if (latkml005.Backend.lower() == "pytorch"):
        if (modelName == "anime"):
            return Informative_Drawings_PyTorch("checkpoints/anime_style/netG_A_latest.pth")
        elif (modelName == "contour"):
            return Informative_Drawings_PyTorch("checkpoints/contour_style/netG_A_latest.pth")
        elif (modelName == "opensketch"):
            return Informative_Drawings_PyTorch("checkpoints/opensketch_style/netG_A_latest.pth")
        elif (modelName == "pxp_001"):
            return Pix2Pix_PyTorch("checkpoints/pix2pix002-001_60_net_G.pth")
        elif (modelName == "pxp_002"):
            return Pix2Pix_PyTorch("checkpoints/pix2pix002-002_60_net_G.pth")
        elif (modelName == "pxp_003"):
            return Pix2Pix_PyTorch("checkpoints/pix2pix002-003_60_net_G.pth")
        elif (modelName == "pxp_004"):
            return Pix2Pix_PyTorch("checkpoints/pix2pix002-004_60_net_G.pth")
        else:
            return None
    else:
        if (modelName == "anime"):
            return Informative_Drawings_Onnx("onnx/anime_style_512x512_simplified.onnx")
        elif (modelName == "contour"):
            return Informative_Drawings_Onnx("onnx/contour_style_512x512_simplified.onnx")
        elif (modelName == "opensketch"):
            return Informative_Drawings_Onnx("onnx/opensketch_style_512x512_simplified.onnx")
        elif (modelName == "pxp_001"):
            return Pix2Pix_Onnx("onnx/pix2pix004-002_140_net_G_simplified.onnx")
        elif (modelName == "pxp_002"):
            return Pix2Pix_Onnx("onnx/pix2pix003-002_140_net_G_simplified.onnx")
        elif (modelName == "pxp_003"):
            return Pix2Pix_Onnx("onnx/neuralcontours_140_net_G_simplified.onnx")
        elif (modelName == "pxp_004"):
            return Pix2Pix_Onnx("onnx/neuralcontours_140_net_G_simplified.onnx")
        else:
            return None

def doInference003(net, verts, dims=256, seqMin=0.0, seqMax=1.0):
    latkml005 = bpy.context.scene.latkml005_settings
    
    bv = vertsToBinvox(verts, dims, doFilter=latkml005.do_filter)
    h5 = binvoxToH5(bv, dims=dims)
    writeTempH5(h5)

    fake_B = net.detect()

    writeTempBinvox(fake_B, dims=dims)
    verts = readTempBinvox(dims=dims)
    dims_ = float(dims - 1)

    for i in range(0, len(verts)):
        x = lb.remap(verts[i][0], 0.0, dims_, seqMin, seqMax)
        y = lb.remap(verts[i][1], 0.0, dims_, seqMin, seqMax)
        z = lb.remap(verts[i][2], 0.0, dims_, seqMin, seqMax)
        verts[i] = Vector((x, y, z))

    return verts

# https://blender.stackexchange.com/questions/262742/python-bpy-2-8-render-directly-to-matrix-array
# https://blender.stackexchange.com/questions/2170/how-to-access-render-result-pixels-from-python-script/3054#3054
def doInference004(net1, net2=None):
    latkml005 = bpy.context.scene.latkml005_settings

    img_np = None
    img_cv = None
    if (latkml005.SourceImage.lower() == "depth"):
        img_np = renderToNp(depthPass=True) # inference expects np array
        img_temp = renderToNp()
        img_cv = npToCv(img_temp) # cv converted image used for color pixels later
    else:
        img_np = renderToNp() # inference expects np array
        img_cv = npToCv(img_np) # cv converted image used for color pixels later

    result = net1.detect(img_np)

    if (net2 != None):
        result = net2.detect(result)

    outputUrl = os.path.join(bpy.app.tempdir, "output.png")
    cv2.imwrite(outputUrl, result)

    im0 = cv2.imread(outputUrl)
    im0 = cv2.bitwise_not(im0) # invert
    imWidth = len(im0[0])
    imHeight = len(im0)
    im = (im0[:,:,0] > latkml005.lineThreshold).astype(np.uint8)
    im = skeletonize(im).astype(np.uint8)
    polys = from_numpy(im, latkml005.csize, latkml005.maxIter)

    laFrame = latk.LatkFrame(frame_number=bpy.context.scene.frame_current)

    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    frame = camera.data.view_frame(scene=bpy.context.scene)
    topRight = frame[0]
    bottomRight = frame[1]
    bottomLeft = frame[2]
    topLeft = frame[3]

    resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))
    xRange = np.linspace(topLeft[0], topRight[0], resolutionX)
    yRange = np.linspace(topLeft[1], bottomLeft[1], resolutionY)

    originalStrokes = []
    originalStrokeColors = []
    separatedStrokes = []
    separatedStrokeColors = []

    # raycasting needs cursor at world origin
    origCursorLocation = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    
    for target in bpy.data.objects:
        if target.type == "MESH":
            matrixWorld = target.matrix_world
            matrixWorldInverted = matrixWorld.inverted()
            origin = matrixWorldInverted @ camera.matrix_world.translation

            for stroke in polys:
                newStroke = []
                newStrokeColor = []
                for point in stroke:
                    rgbPixel = img_cv[point[1]][point[0]]
                    rgbPixel2 = (rgbPixel[2], rgbPixel[1], rgbPixel[0], 1)

                    xPos = lb.remap(point[0], 0, resolutionX, xRange.min(), xRange.max())
                    yPos = lb.remap(point[1], 0, resolutionY, yRange.max(), yRange.min())
                   
                    pixelVector = Vector((xPos, yPos, topLeft[2]))
                    pixelVector.rotate(camera.matrix_world.to_quaternion())
                    destination = matrixWorldInverted @ (pixelVector + camera.matrix_world.translation) 
                    direction = (destination - origin).normalized()
                    hit, location, norm, face = target.ray_cast(origin, direction)

                    if hit:
                        location = target.matrix_world @ location
                        co = (location.x, location.y, location.z)
                        newStroke.append(co)
                        newStrokeColor.append(rgbPixel2)

                if (len(newStroke) > 1):
                    originalStrokes.append(newStroke)
                    originalStrokeColors.append(newStrokeColor)

        for i in range(0, len(originalStrokes)):
            separatedTempStrokes, separatedTempStrokeColors = lb.separatePointsByDistance(originalStrokes[i], originalStrokeColors[i], latkml005.distThreshold)

            for j in range(0, len(separatedTempStrokes)):
                separatedStrokes.append(separatedTempStrokes[j])
                separatedStrokeColors.append(separatedTempStrokeColors[j])

        for i in range(0, len(separatedStrokes)):
            laPoints = []
            for j in range(0, len(separatedStrokes[i])):
                laPoint = latk.LatkPoint(separatedStrokes[i][j])
                laPoint.vertex_color = separatedStrokeColors[i][j]
                laPoints.append(laPoint)

            if (len(laPoints) > 1):
                laFrame.strokes.append(latk.LatkStroke(laPoints))

    bpy.context.scene.cursor.location = origCursorLocation
    return laFrame

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

def createOnnxNetwork(modelPath):
    modelPath = getModelPath(modelPath)
    net = None

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL    
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    
    #if (ort.get_device().lower() == "gpu"):
    net = ort.InferenceSession(modelPath, so, providers=["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"])
    #else:
        #net = ort.InferenceSession(modelPath, so)

    return net


class Informative_Drawings_Onnx():
    def __init__(self, modelPath):       
        self.net = createOnnxNetwork(modelPath)
        
        input_shape = self.net.get_inputs()[0].shape
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(img.astype(np.float32), (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run([self.output_name], {self.input_name: blob})

        result = outs[0].squeeze()
        result *= 255
        result = cv2.resize(result.astype('uint8'), (srcimg.shape[1], srcimg.shape[0]))
        return result


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/1113
class Pix2Pix_Onnx():
    def __init__(self, modelPath):
        self.net = createOnnxNetwork(modelPath)

        self.input_size = 256
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
        print("input_name = " + self.input_name)
        print("output_name = " + self.output_name)

    def detect(self, srcimg):
        if isinstance(srcimg, str):
            srcimg=cv2.imdecode(np.fromfile(srcimg, dtype=np.uint8), -1)
        elif isinstance(srcimg, np.ndarray):
            srcimg=srcimg.copy()
        # srcimg=srcimg[0:256, 0:256]
        img = cv2.resize(srcimg, (self.input_size, self.input_size))
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        #input_image = input_image / 255.0
        input_image = (input_image - 0.5) / 0.5 
        input_image = input_image.astype('float32')
        print(input_image.shape)
        # x = x[None,:,:,:]
        outs = self.net.run(None, {self.input_name: input_image})[0].squeeze(axis=0)
        outs = np.clip(((outs*0.5+0.5) * 255), 0, 255).astype(np.uint8) 
        outs = outs.transpose(1, 2, 0).astype('uint8')
        outs = cv2.cvtColor(outs, cv2.COLOR_RGB2BGR)
        outs = np.hstack((img, outs))
        print("outs",outs.shape)
        
        # [y:y+height, x:x+width]
        outs = outs[0:256, 256:512]
        return cv2.resize(outs, (srcimg.shape[1], srcimg.shape[0]))


def getPyTorchDevice(mps=True):
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and mps==True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def createPyTorchNetwork(modelPath, net_G, device): #, input_nc=3, output_nc=1, n_blocks=3):
    #device = getPyTorchDevice()
    modelPath = getModelPath(modelPath)
    net_G.to(device)
    net_G.load_state_dict(torch.load(modelPath, map_location=device))
    net_G.eval()
    return net_G


class Informative_Drawings_PyTorch():
    def __init__(self, modelPath):
        self.device = getPyTorchDevice()         
        generator = Generator(3, 1, 3) # input_nc=3, output_nc=1, n_blocks=3
        self.net_G = createPyTorchNetwork(modelPath, generator, self.device)   

    def detect(self, srcimg):
        with torch.no_grad():   
            srcimg2 = np.transpose(srcimg, (2, 0, 1))

            tensor_array = torch.from_numpy(srcimg2)
            input_tensor = tensor_array.to(self.device)
            output_tensor = self.net_G(input_tensor)

            result = output_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            result *= 255
            result = cv2.resize(result, (srcimg.shape[1], srcimg.shape[0]))
            
            return result


class Pix2Pix_PyTorch():
    def __init__(self, modelPath):
        self.device = getPyTorchDevice() 
        
        Opt = namedtuple("Opt", ["model","gpu_ids","isTrain","checkpoints_dir","name","preprocess","input_nc","output_nc","ngf","netG","norm","no_dropout","init_type", "init_gain","load_iter","dataset_mode","epoch"])
        opt = Opt("pix2pix", [], False, "", "", False, 3, 3, 64, "unet_256", "batch", True, "normal", 0.02, 0, "aligned", "latest")

        generator = pix2pix_model.Pix2PixModel(opt).netG 

        self.net_G = createPyTorchNetwork(modelPath, generator, self.device)   

    def detect(self, srcimg):
        with torch.no_grad():  
            srcimg2 = cv2.resize(srcimg, (256, 256))
            input_image = cv2.cvtColor(srcimg2, cv2.COLOR_BGR2RGB)
            input_image = input_image.transpose(2, 0, 1)
            input_image = np.expand_dims(input_image, axis=0)
            #input_image = input_image / 255.0
            input_image = (input_image - 0.5) / 0.5 
            input_image = input_image.astype('float32')

            tensor_array = torch.from_numpy(input_image)
            input_tensor = tensor_array.to(self.device)
            output_tensor = self.net_G(input_tensor)

            result = output_tensor[0].detach().cpu().numpy() #.transpose(1, 2, 0)
            result = np.clip(((result*0.5+0.5) * 255), 0, 255) #.astype(np.uint8) 
            result = result.transpose(1, 2, 0) #.astype('uint8')
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            #result = output_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            #result *= 255
            
            result = cv2.resize(result, (srcimg.shape[1], srcimg.shape[0]))
            
            return result


class Vox2Vox_PyTorch():
    def __init__(self, modelPath):
        self.device = getPyTorchDevice(mps=False) # MPS needs to support operator aten::slow_conv3d_forward          
        generator = GeneratorUNet()
        if self.device.type == "cuda":
            generator = generator.cuda()

        self.net_G = createPyTorchNetwork(modelPath, generator, self.device)

        self.transforms_ = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self):
        Tensor = None
        if self.device.type == "cuda":
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor

        val_dataloader = DataLoader(
            CTDataset(bpy.app.tempdir, transforms_=self.transforms_, isTest=True),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        dataiter = iter(val_dataloader)
        imgs = next(dataiter) #dataiter.next()

        """Saves a generated sample from the validation set"""
        real_A = Variable(imgs["A"].unsqueeze_(1).type(Tensor))
        #real_B = Variable(imgs["B"].unsqueeze_(1).type(Tensor))
        fake_B = self.net_G(real_A)

        return fake_B.cpu().detach().numpy()


def doVoxelOpCore(context, allFrames=False):
    latkml005 = context.scene.latkml005_settings

    dims = None
    
    op1 = latkml005.Operation1.lower() 
    op2 = latkml005.Operation2.lower() 
    op3 = latkml005.Operation3.lower() 

    net1 = None
    obj = lb.ss()
    la = lb.latk.Latk(init=True)
    gp = lb.fromLatkToGp(la, resizeTimeline = False)

    start = bpy.context.scene.frame_current
    end = start + 1
    if (allFrames == True):
        start, end = lb.getStartEnd()
    #if (op1 != "none"):
        #start = start - 1

    for i in range(start, end):
        lb.goToFrame(i)

        origCursorLocation = bpy.context.scene.cursor.location
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

        #lb.s(obj)
        #bpy.context.view_layer.objects.active = obj
        #bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

        #verts, colors = lb.getVertsAndColors(target=obj, useWorldSpace=False, useColors=True, useBmesh=False)
        verts, colors = lb.getVertices(obj, getColors=True, worldSpace=False)
        #verts = lb.getVertices(obj)
        faces = lb.getFaces(obj)
        matrix_world = obj.matrix_world
        
        #bounds = obj.dimensions
        seqAbs = None #(bounds.x + bounds.y + bounds.z) / 3.0

        seqMin = 0.0
        seqMax = 1.0
    
        for vert in verts:
            x = vert[0]
            y = vert[1]
            z = vert[2]
            if (x < seqMin):
                seqMin = x
            if (x > seqMax):
                seqMax = x
            if (y < seqMin):
                seqMin = y
            if (y > seqMax):
                seqMax = y
            if (z < seqMin):
                seqMin = z
            if (z > seqMax):
                seqMax = z

        seqAbs = abs(seqMax - seqMin)

        if (op1 != "none"):
            if not net1:
                net1 = loadModel003()    
                dims = latkml005.dims   

            avgPosOrig = None
            if (latkml005.do_recenter == True):
                avgPosOrig = getAveragePosition(verts)

            vertsOrig = np.array(verts) #.copy()
            verts = doInference003(net1, verts, dims, seqMin, seqMax)

            if (latkml005.do_recenter == True):
                avgPosNew = getAveragePosition(verts)
                diffPos = avgPosOrig - avgPosNew
                for i in range(0, len(verts)):
                    verts[i] = verts[i] + diffPos

            colors = transferVertexColors(vertsOrig, colors, verts)

        if (op2 == "get_edges" and op1 == "none"):
            vertsOrig = np.array(verts)
            verts = differenceEigenvalues(verts)
            colors = transferVertexColors(vertsOrig, colors, verts)           

        bpy.context.scene.cursor.location = origCursorLocation

        #gp = None

        if (op3 == "skel_gen" and op1 == "none"):
            skelGen(verts, faces, matrix_world=matrix_world)
        elif (op3 == "contour_gen" and op1 == "none"):
            contourGen(verts, faces, matrix_world=matrix_world)
        else:
            strokeGen(verts, colors, matrix_world=matrix_world, radius=seqAbs * latkml005.strokegen_radius, minPointsCount=latkml005.strokegen_minPointsCount, origin=obj.location) #limitPalette=context.scene.latk_settings.paletteLimit)

def group_points_into_strokes(points, radius, minPointsCount):
    strokeGroups = []
    unassigned_points = set(range(len(points)))

    while len(unassigned_points) > 0:
        strokeGroup = [next(iter(unassigned_points))]
        unassigned_points.remove(strokeGroup[0])

        for i in range(len(points)):
            if i in unassigned_points and cdist([points[i]], [points[strokeGroup[-1]]])[0][0] < radius:
                strokeGroup.append(i)
                unassigned_points.remove(i)

        if (len(strokeGroup) >= minPointsCount):
            strokeGroups.append(strokeGroup)

        print("Found " + str(len(strokeGroups)) + " strokeGroups, " + str(len(unassigned_points)) + " points remaining.")
    return strokeGroups

def strokeGen(verts, colors, matrix_world=None, radius=2, minPointsCount=5, origin=None): #limitPalette=32):
    latkml005 = bpy.context.scene.latkml005_settings
    origCursorLocation = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    gp = lb.getActiveGp()
    layer = lb.getActiveLayer()
    if not layer:
        layer = gp.data.layers.new(name="meshToGp")
    frame = lb.getActiveFrame()
    if not frame or frame.frame_number != lb.currentFrame():
        frame = layer.frames.new(lb.currentFrame())

    strokeGroups = group_points_into_strokes(verts, radius, minPointsCount)

    lastColor = (1,1,1,1)
    for strokeGroup in strokeGroups:
        strokeColors = []
        for i in range(0, len(strokeGroup)):
            try:
                newColor = colors[strokeGroup[i]]
                strokeColors.append(newColor)
                lastColor = newColor
            except:
                strokeColors.append((0,1,0,1)) #lastColor)
        '''
        if (limitPalette == 0):
            lb.createColor(color)
        else:
            lb.createAndMatchColorPalette(color, limitPalette, 5) # num places
        '''

        stroke = frame.strokes.new()
        stroke.display_mode = '3DSPACE'
        stroke.line_width = int(latkml005.thickness) #10 # adjusted from 100 for 2.93
        stroke.material_index = gp.active_material_index

        stroke.points.add(len(strokeGroup))

        for i, strokeIndex in enumerate(strokeGroup):    
            if not matrix_world:
                point = verts[strokeIndex]
            else:
                point = matrix_world @ Vector(verts[strokeIndex])

            #point = matrixWorldInverted @ Vector((point[0], point[2], point[1]))
            #point = (point[0], point[1], point[2])
            pressure = 1.0
            strength = 1.0
            lb.createPoint(stroke, i, point, pressure, strength, strokeColors[i])

    bpy.context.scene.cursor.location = origCursorLocation
    return gp

def contourGen(verts, faces, matrix_world):
    latkml005 = bpy.context.scene.latkml005_settings
    origCursorLocation = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    la = latk.Latk(init=True)

    gp = lb.getActiveGp()
    layer = lb.getActiveLayer()
    if not layer:
        layer = gp.data.layers.new(name="meshToGp")
    frame = lb.getActiveFrame()
    if not frame or frame.frame_number != lb.currentFrame():
        frame = layer.frames.new(lb.currentFrame())

    mesh = None

    try: 
        mesh = trimesh.Trimesh(verts, faces)
    except:
        tri = Delaunay(verts)
        mesh = trimesh.Trimesh(tri.points, tri.simplices)

    bounds = lb.getDistance(mesh.bounds[0], mesh.bounds[1])

    # generate a set of contour lines at regular intervals
    interval = bounds * 0.01 #0.03  #0.1 # the spacing between contours
    print("Interval: " + str(interval))

    # x, z
    slice_range = np.arange(mesh.bounds[0][2], mesh.bounds[1][2], interval)
    # y
    #slice_range = np.arange(mesh.bounds[0][1], mesh.bounds[0][2], interval)

    # loop over the z values and generate a contour at each level
    for slice_pos in slice_range:
        # x
        #slice_mesh = mesh.section(plane_origin=[slice_pos, 0, 0], plane_normal=[1, 0, 0])
        # y
        #slice_mesh = mesh.section(plane_origin=[0, slice_pos, 0], plane_normal=[0, 1, 0])
        # z
        slice_mesh = mesh.section(plane_origin=[0, 0, slice_pos], plane_normal=[0, 0, 1])
        
        if slice_mesh != None:
            for entity in slice_mesh.entities:
                stroke = frame.strokes.new()
                stroke.display_mode = '3DSPACE'
                stroke.line_width = int(latkml005.thickness) #10 # adjusted from 100 for 2.93
                stroke.material_index = gp.active_material_index
                stroke.points.add(len(entity.points))

                for i, index in enumerate(entity.points):
                    vert = None
                    if not matrix_world:
                        vert = slice_mesh.vertices[index] 
                    else:
                        vert = matrix_world @ Vector(slice_mesh.vertices[index])
                    #vert = [vert[0], vert[1], vert[2]]
                    lb.createPoint(stroke, i, vert, 1.0, 1.0)

    #lb.fromLatkToGp(la, resizeTimeline=False)
    #lb.setThickness(latkml005.thickness)

    bpy.context.scene.cursor.location = origCursorLocation
    return gp

def skelGen(verts, faces, matrix_world):
    latkml005 = bpy.context.scene.latkml005_settings
    origCursorLocation = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    la = latk.Latk(init=True)

    gp = lb.getActiveGp()
    layer = lb.getActiveLayer()
    if not layer:
        layer = gp.data.layers.new(name="meshToGp")
    frame = lb.getActiveFrame()
    if not frame or frame.frame_number != lb.currentFrame():
        frame = layer.frames.new(lb.currentFrame())

    mesh = None

    try: 
        mesh = trimesh.Trimesh(verts, faces)
    except:
        tri = Delaunay(verts)
        mesh = trimesh.Trimesh(tri.points, tri.simplices)

    fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
    skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)

    for entity in skel.skeleton.entities:
        stroke = frame.strokes.new()
        stroke.display_mode = '3DSPACE'
        stroke.line_width = int(latkml005.thickness) #10 # adjusted from 100 for 2.93
        stroke.material_index = gp.active_material_index
        stroke.points.add(len(entity.points))

        for i, index in enumerate(entity.points):
            vert = None
            if not matrix_world:
                vert = skel.vertices[index]
            else:
                vert = matrix_world @ Vector(skel.vertices[index])
            lb.createPoint(stroke, i, vert, 1.0, 1.0)

    #lb.fromLatkToGp(la, resizeTimeline=False)
    #lb.setThickness(latkml005.thickness)

    bpy.context.scene.cursor.location = origCursorLocation
    return gp

def differenceEigenvalues(verts):
    # MIT License Copyright (c) 2015 Dena Bazazian Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    pdVerts = pd.DataFrame(verts, columns=["x", "y", "z"])
    pcd1 = PyntCloud(pdVerts)
        
    # define hyperparameters
    k_n = 50 # 50
    thresh = 0.03 # 0.03

    pcd_np = np.zeros((len(pcd1.points),6))

    # find neighbors
    kdtree_id = pcd1.add_structure("kdtree")
    k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id) 

    # calculate eigenvalues
    ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = pcd1.points['x'].values 
    y = pcd1.points['y'].values 
    z = pcd1.points['z'].values 

    e1 = pcd1.points['e3('+str(k_n+1)+')'].values
    e2 = pcd1.points['e2('+str(k_n+1)+')'].values
    e3 = pcd1.points['e1('+str(k_n+1)+')'].values

    sum_eg = np.add(np.add(e1,e2),e3)
    sigma = np.divide(e1,sum_eg)
    sigma_value = sigma

    # visualize the edges
    sigma = sigma>thresh

    # Save the edges and point cloud
    thresh_min = sigma_value < thresh
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > thresh
    sigma_value[thresh_max] = 255

    pcd_np[:,0] = x
    pcd_np[:,1] = y
    pcd_np[:,2] = z
    pcd_np[:,3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:,3] == 0), axis=0) 
    print(len(edge_np))

    clmns = ['x','y','z','red','green','blue']
    #pcd_pd = pd.DataFrame(data=pcd_np,columns=clmns)
    #pcd_pd['red'] = sigma_value.astype(np.uint8)

    #pcd_points = PyntCloud(pcd_pd)
    #edge_points = PyntCloud(pd.DataFrame(data=edge_np,columns=clmns))

    #PyntCloud.to_file(edge_points, outputPath) # Save just the edge points
    newVerts = []
    #for i in range(0, len(edge_points.points)):
    #    newVerts.append((edge_points.points["x"][i], edge_points.points["y"][i], edge_points.points["z"][i]))
    for edge in edge_np:
        newVerts.append((edge[0], edge[1], edge[2]))

    return newVerts

def strokeGen_orig(obj=None, strokeLength=1, strokeGaps=10.0, shuffleOdds=1.0, spreadPoints=0.1, limitPalette=32):
    if not obj:
        obj = lb.ss()
    mesh = obj.data
    mat = obj.matrix_world
    #~
    gp = lb.getActiveGp()
    layer = lb.getActiveLayer()
    if not layer:
        layer = gp.data.layers.new(name="meshToGp")
    frame = lb.getActiveFrame()
    if not frame or frame.frame_number != lb.currentFrame():
        frame = layer.frames.new(lb.currentFrame())
    #~
    images = None
    try:
        images = lb.getUvImages()
    except:
        pass
    #~
    allPoints, allColors = lb.getVertsAndColorsAlt(target=obj, useWorldSpace=True, useColors=True, useBmesh=False)
    #~
    pointSeqsToAdd = []
    colorsToAdd = []
    for i in range(0, len(allPoints), strokeLength):
        color = None
        if not images:
            try:
                color = allColors[i]
            except:
                color = lb.getColorExplorer(obj, i)
        else:
            try:
                color = lb.getColorExplorer(obj, i, images)
            except:
                color = lb.getColorExplorer(obj, i)
        colorsToAdd.append(color)
        #~
        pointSeq = []
        for j in range(0, strokeLength):
            #point = allPoints[i]
            try:
                point = allPoints[i+j]
                if (len(pointSeq) == 0 or lb.getDistance(pointSeq[len(pointSeq)-1], point) < strokeGaps):
                    pointSeq.append(point)
            except:
                break
        if (len(pointSeq) > 0): 
            pointSeqsToAdd.append(pointSeq)
    for i, pointSeq in enumerate(pointSeqsToAdd):
        color = colorsToAdd[i]
        #createColor(color)
        if (limitPalette == 0):
            lb.createColor(color)
        else:
            lb.createAndMatchColorPalette(color, limitPalette, 5) # num places
        #stroke = frame.strokes.new(getActiveColor().name)
        #stroke.draw_mode = "3DSPACE"
        stroke = frame.strokes.new()
        stroke.display_mode = '3DSPACE'
        stroke.line_width = 10 # adjusted from 100 for 2.93
        stroke.material_index = gp.active_material_index

        stroke.points.add(len(pointSeq))

        if (random.random() < shuffleOdds):
            random.shuffle(pointSeq)

        for j, point in enumerate(pointSeq):    
            x = point[0] + (random.random() * 2.0 * spreadPoints) - spreadPoints
            y = point[2] + (random.random() * 2.0 * spreadPoints) - spreadPoints
            z = point[1] + (random.random() * 2.0 * spreadPoints) - spreadPoints
            pressure = 1.0
            strength = 1.0
            lb.createPoint(stroke, j, (x, y, z), pressure, strength)

def scale_numpy_array(arr, min_v, max_v):
    new_range = (min_v, max_v)
    max_range = max(new_range)
    min_range = min(new_range)

    scaled_unit = (max_range - min_range) / (np.max(arr) - np.min(arr))
    return arr * scaled_unit - np.min(arr) * scaled_unit + min_range

def resizeVoxels(voxel, shape):
    ratio = shape[0] / voxel.shape[0]
    voxel = nd.zoom(voxel,
            ratio,
            order=1, 
            mode='nearest')
    voxel[np.nonzero(voxel)] = 1.0
    return voxel

def getAveragePositionObj(obj=None, applyTransforms=False):
    if not obj:
        obj = lb.ss()
    if (applyTransforms == True):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return getAveragePosition(obj.data.vertices, obj.matrix_world)

def getAveragePosition(verts, matrix_world=None):
    returns = Vector((0,0,0))
    for vert in verts:
        if not matrix_world:
            returns += Vector(vert)
        else:
            returns += matrix_world @ Vector(vert)
    returns /= float(len(verts))
    return returns

def transferVertexColors(sourceVerts, sourceColors, destVerts):
    sourceVerts = np.array(sourceVerts)
    sourceColors = np.array(sourceColors)
    destVerts = np.array(destVerts)

    tree = cKDTree(sourceVerts)

    _, indices = tree.query(destVerts) #, k=1)

    destColors = sourceColors[indices]

    return destColors

def vertsToBinvox(verts, dims=256, doFilter=False, axis='xyz'):
    shape = (dims, dims, dims)
    data = np.zeros(shape, dtype=bool)
    translate = (0, 0, 0)
    scale = 1
    axis_order = axis
    bv = binvox_rw.Voxels(data, shape, translate, scale, axis_order)

    verts = lb.normalize(verts, minVal=0.0, maxVal=float(dims-1))
    
    for vert in verts:
        x = int(vert[0])
        y = int(vert[1])
        z = int(vert[2])
        data[x][y][z] = True

    if (doFilter == True):
        for i in range(0, 1): # 1
            nd.binary_dilation(bv.data.copy(), output=bv.data)

        for i in range(0, 3): # 3
            nd.sobel(bv.data.copy(), output=bv.data)

        nd.median_filter(bv.data.copy(), size=4, output=bv.data) # 4

        for i in range(0, 2): # 2
            nd.laplace(bv.data.copy(), output=bv.data)

        for i in range(0, 0): # 0
            nd.binary_erosion(bv.data.copy(), output=bv.data)

    return bv

'''
def binvoxToVerts(voxel, dims=256, axis='xyz'):
    verts = []
    for x in range(0, dims):
        for y in range(0, dims):
            for z in range(0, dims):
                if (voxel.data[x][y][z] == True):
                    verts.append([x, y, z])
    return verts
'''

def binvoxToH5(voxel, dims=256):
    shape=(dims, dims, dims)   
    voxel_data = voxel.data.astype(float) #voxel.data.astype(np.float)
    if shape is not None and voxel_data.shape != shape:
        voxel_data = resize(voxel.data.astype(np.float64), shape)
    return voxel_data

def h5ToBinvox(data, dims=256):
    data = np.rint(data).astype(np.uint8)
    shape = (dims, dims, dims) #data.shape
    translate = [0, 0, 0]
    scale = 1.0
    axis_order = 'xzy'
    return binvox_rw.Voxels(data, shape, translate, scale, axis_order)

def writeTempH5(data):
    url = os.path.join(bpy.app.tempdir, "output.im")
    f = h5py.File(url, 'w')
    # more compression options: https://docs.h5py.org/en/stable/high/dataset.html
    f.create_dataset('data', data=data, compression='gzip')
    f.flush()
    f.close()

def readTempH5():
    url = os.path.join(bpy.app.tempdir, "output.im")
    return h5py.File(url, 'r').get('data')[()]

def writeTempBinvox(data, dims=256):
    url = os.path.join(bpy.app.tempdir, "output.binvox")
    data = np.rint(data).astype(np.uint8)
    shape = (dims, dims, dims) #data.shape
    translate = [0, 0, 0]
    scale = 1.0
    axis_order = 'xzy'
    voxel = binvox_rw.Voxels(data, shape, translate, scale, axis_order)

    with open(url, 'bw') as f:
        voxel.write(f)

def readTempBinvox(dims=256, axis='xyz'):
    url = os.path.join(bpy.app.tempdir, "output.binvox")
    voxel = None
    print("Reading from: " + url)
    with open(url, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f, True) # fix coords
    verts = []
    for x in range(0, dims):
        for y in range(0, dims):
            for z in range(0, dims):
                if (voxel.data[x][y][z] == True):
                    verts.append([z, y, x])
    return verts