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
from mathutils import Vector, Matrix
import bmesh

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
import random
import itertools

import h5py
import skeletor as sk
import trimesh
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import scipy.ndimage as nd
from pyntcloud import PyntCloud 
import pandas as pd
import pdb

from .skeleton_tracing.swig.trace_skeleton import *

#from . import binvox_rw
from .vox2vox import binvox_rw

from . latk_ml import *
from . latk_pytorch import *
from . latk_onnx import *


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
        #row = box.row()
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

        row = box.row()
        row.prop(latkml005, "Backend")

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        box = layout.box()

        row = box.row()
        row.operator("latkml005_button.singleframe003")
        row.operator("latkml005_button.allframes003")

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
    return Vox2Vox_PyTorch(__name__, "model/" + modelName + ".pth")

def modelSelector004(modelName):
    modelName = modelName.lower()
    latkml005 = bpy.context.scene.latkml005_settings

    if (latkml005.Backend.lower() == "pytorch"):
        if (modelName == "anime"):
            return Informative_Drawings_PyTorch(__name__, "checkpoints/anime_style/netG_A_latest.pth")
        elif (modelName == "contour"):
            return Informative_Drawings_PyTorch(__name__, "checkpoints/contour_style/netG_A_latest.pth")
        elif (modelName == "opensketch"):
            return Informative_Drawings_PyTorch(__name__, "checkpoints/opensketch_style/netG_A_latest.pth")
        elif (modelName == "pxp_001"):
            return Pix2Pix_PyTorch(__name__, "checkpoints/pix2pix002-001_60_net_G.pth")
        elif (modelName == "pxp_002"):
            return Pix2Pix_PyTorch(__name__, "checkpoints/pix2pix002-002_60_net_G.pth")
        elif (modelName == "pxp_003"):
            return Pix2Pix_PyTorch(__name__, "checkpoints/pix2pix002-003_60_net_G.pth")
        elif (modelName == "pxp_004"):
            return Pix2Pix_PyTorch(__name__, "checkpoints/pix2pix002-004_60_net_G.pth")
        else:
            return None
    else:
        if (modelName == "anime"):
            return Informative_Drawings_Onnx(__name__, "onnx/anime_style_512x512_simplified.onnx")
        elif (modelName == "contour"):
            return Informative_Drawings_Onnx(__name__, "onnx/contour_style_512x512_simplified.onnx")
        elif (modelName == "opensketch"):
            return Informative_Drawings_Onnx(__name__, "onnx/opensketch_style_512x512_simplified.onnx")
        elif (modelName == "pxp_001"):
            return Pix2Pix_Onnx(__name__, "onnx/pix2pix004-002_140_net_G_simplified.onnx")
        elif (modelName == "pxp_002"):
            return Pix2Pix_Onnx(__name__, "onnx/pix2pix003-002_140_net_G_simplified.onnx")
        elif (modelName == "pxp_003"):
            return Pix2Pix_Onnx(__name__, "onnx/neuralcontours_140_net_G_simplified.onnx")
        elif (modelName == "pxp_004"):
            return Pix2Pix_Onnx(__name__, "onnx/neuralcontours_140_net_G_simplified.onnx")
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

