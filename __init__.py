bl_info = {
    "name": "latk_ml_005", 
    "author": "Nick Fox-Gieg",
    "version": (0, 0, 1),
    "blender": (3, 0, 0),
    "description": "Generate brushstrokes from a mesh using informative-drawings and pix2pix",
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


class latkml005_Button_AllFrames(bpy.types.Operator):
    """Operate on all frames"""
    bl_idname = "latkml005_button.allframes"
    bl_label = "All Frames"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = loadModel()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())

        start, end = lb.getStartEnd()
        for i in range(start, end):
            lb.goToFrame(i)
            laFrame = doInference(net1, net2)
            la.layers[0].frames.append(laFrame)

        lb.fromLatkToGp(la, resizeTimeline=False)
        lb.setThickness(latkml005.thickness)
        return {'FINISHED'}


class latkml005_Button_SingleFrame(bpy.types.Operator):
    """Operate on a single frame"""
    bl_idname = "latkml005_button.singleframe"
    bl_label = "Single Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml005 = context.scene.latkml005_settings
        net1, net2 = loadModel()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())
        laFrame = doInference(net1, net2)
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
        row.operator("latkml005_button.singleframe")
        row.operator("latkml005_button.allframes")

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


classes = (
    OBJECT_OT_latkml005_prefs,
    latkml005Preferences,
    latkml005Properties,
    latkml005Properties_Panel,
    latkml005_Button_AllFrames,
    latkml005_Button_SingleFrame
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

def loadModel():
    latkml005 = bpy.context.scene.latkml005_settings
   
    returns1 = modelSelector(latkml005.ModelStyle1)
    returns2 = modelSelector(latkml005.ModelStyle2)

    return returns1, returns2

def modelSelector(modelName):
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

# https://blender.stackexchange.com/questions/262742/python-bpy-2-8-render-directly-to-matrix-array
# https://blender.stackexchange.com/questions/2170/how-to-access-render-result-pixels-from-python-script/3054#3054
def doInference(net1, net2=None):
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
    net = ort.InferenceSession(modelPath, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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


def getPyTorchDevice():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
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