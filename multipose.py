
# Shamelessly based on the Javascript version from:
# https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
# 
# OBS 1: 
# Near to zero efforts were made to optimize the code. 
# Could we get 1000x as in https://youtu.be/nxWginnBklU ??? :)
#

import numpy as np
import scipy.ndimage as ndi
from PIL import ImageDraw, ImageOps

MOBILENET_V1_CONFIG = {
    'architecture': 'MobileNetV1',
    'outputStride': 16,
    'multiplier': 0.75,
    'inputResolution': 257
    }

partNames = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
]
NUM_KEYPOINTS = len(partNames)
partIds = {k:i for i,k in enumerate(partNames)}

partIds2Names = {i:k for i,k in enumerate(partNames)}

connectedPartNames = [
        ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
        ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
        ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
        ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
        ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
    ]
poseChain = [
        ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
        ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
        ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
        ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
        ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
        ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
        ['rightKnee', 'rightAnkle']
    ]

partChannels = [
        'left_face',
        'right_face',
        'right_upper_leg_front',
        'right_lower_leg_back',
        'right_upper_leg_back',
        'left_lower_leg_front',
        'left_upper_leg_front',
        'left_upper_leg_back',
        'left_lower_leg_back',
        'right_feet',
        'right_lower_leg_front',
        'left_feet',
        'torso_front',
        'torso_back',
        'right_upper_arm_front',
        'right_upper_arm_back',
        'right_lower_arm_back',
        'left_lower_arm_front',
        'left_upper_arm_front',
        'left_upper_arm_back',
        'left_lower_arm_back',
        'right_hand',
        'right_lower_arm_front',
        'left_hand'
    ];

parentChildrenTuples = [[partIds[parentJoinName], partIds[childJoinName]] for parentJoinName,childJoinName in poseChain]

parentToChildEdges = [i[1] for i in parentChildrenTuples]

childToParentEdges = [i[0] for i in parentChildrenTuples]

def getOffsetPoint(y, x, keypoint, offsets):
    return {
        'y': offsets[y, x, keypoint],
        'x': offsets[y, x, keypoint + NUM_KEYPOINTS]
    }

def getImageCoords(part, outputStride, offsets):
#     heatmapY = part['heatmapY']
#     heatmapX = part['heatmapX']
#     keypoint = part['id']
#     _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets)
#     y = _a['y']
#     x = _a['x']
    return {
        'x': part['heatmapX'] * outputStride + offsets[part['heatmapY'], part['heatmapX'], part['id'] + NUM_KEYPOINTS],
        'y': part['heatmapY'] * outputStride + offsets[part['heatmapY'], part['heatmapX'], part['id']]
    }


def squaredDistance(y1, x1, y2, x2):
    return (x2 - x1)**2 + (y2 - y1)**2
    
def getDisplacement(edgeId, point, displacements):
    numEdges = int(displacements.shape[2] / 2)
    return {
        'y': displacements[point['y'], point['x'], edgeId],
        'x': displacements[point['y'], point['x'], numEdges + edgeId]
    }

def addVectors(a, b):
    return { 'x': a['x'] + b['x'], 'y': a['y'] + b['y'] };

def clamp(n, smallest, largest): 
    return int(max(smallest, min(n, largest)))

def getStridedIndexNearPoint(point, outputStride, height, width):
    return {
        'y': clamp(point['y'] / outputStride, 0, height - 1),
        'x': clamp(point['x'] / outputStride, 0, width - 1)
    }

# from https://github.com/rwightman/posenet-python
def buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius=1, scores=None):
    parts = []
    num_keypoints = scores.shape[2]
    lmd = 2 * kLocalMaximumRadius + 1

    # NOTE it seems faster to iterate over the keypoints and perform maximum_filter
    # on each subarray vs doing the op on the full score array with size=(lmd, lmd, 1)
    for keypoint_id in range(num_keypoints):
        kp_scores = scores[:, :, keypoint_id].copy()
        kp_scores[kp_scores < scoreThreshold] = 0.
        
        max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode='constant')
        
        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
        max_loc_idx = max_loc.nonzero()
        for y, x in zip(*max_loc_idx):
            parts.append({'score': scores[y, x, keypoint_id], 
                      'part': {'heatmapY': y, 'heatmapX': x, 'id': keypoint_id }
                      })

    return parts



def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets, outputStride, displacements, offsetRefineStep=None):
    if (offsetRefineStep == None):  offsetRefineStep = 2
    height = scoresBuffer.shape[0]
    width = scoresBuffer.shape[1];
    sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypoint['position'], outputStride, height, width)
    displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements)
    displacedPoint = addVectors(sourceKeypoint['position'], displacement)
    targetKeypoint = displacedPoint
    for i in range(offsetRefineStep):
        targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint, outputStride, height, width)
        offsetPoint = getOffsetPoint(targetKeypointIndices['y'], targetKeypointIndices['x'], targetKeypointId, offsets)
        targetKeypoint = addVectors({
            'x': targetKeypointIndices['x'] * outputStride,
            'y': targetKeypointIndices['y'] * outputStride
        }, { 'x': offsetPoint['x'], 'y': offsetPoint['y'] })
    targetKeyPointIndices = getStridedIndexNearPoint(targetKeypoint, outputStride, height, width)
    score = scoresBuffer[targetKeyPointIndices['y'], targetKeyPointIndices['x'], targetKeypointId]
    return { 'position': targetKeypoint, 'score': score }


def decodePose(root, rootImageCoords, scores, offsets, outputStride, displacementsFwd, displacementsBwd):
    numParts = scores.shape[2]
    numEdges = len(parentToChildEdges);
    instanceKeypoints = dict(zip(partNames,[None]*numParts))
    rootPart = root['part']
    rootScore = root['score']
    rootPoint = rootImageCoords #getImageCoords(rootPart, outputStride, offsets)
    instanceKeypoints[partNames[rootPart['id']]] = {
        'score': rootScore,
        'position': rootPoint}

    for edge in range(numEdges)[::-1]:
        sourceKeypointId = partIds2Names[parentToChildEdges[edge]]
        targetKeypointId = partIds2Names[childToParentEdges[edge]]
        if (instanceKeypoints[sourceKeypointId] and (not instanceKeypoints[targetKeypointId])):
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], partIds[targetKeypointId], scores, offsets, outputStride, displacementsBwd)
  
    for edge in range(numEdges):
        sourceKeypointId = partIds2Names[childToParentEdges[edge]]
        targetKeypointId = partIds2Names[parentToChildEdges[edge]]
        if (instanceKeypoints[sourceKeypointId] and (not instanceKeypoints[targetKeypointId])):
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], partIds[targetKeypointId], scores, offsets, outputStride, displacementsFwd)
    
    return instanceKeypoints

def withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, keypointId):
    x = rootImageCoords['x']
    y = rootImageCoords['y']

    for pose in poses:
      keypoints = pose['keypoints']
      correspondingKeypoint = keypoints[partIds2Names[keypointId]]['position']
      if squaredDistance(y, x, correspondingKeypoint['y'], correspondingKeypoint['x']) <= squaredNmsRadius:
        return True
    return False

def getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints):
    notOverlappedKeypointScores = 0.0
    for keypointId,ik in enumerate(instanceKeypoints.values()):
        position = ik['position']
        score = ik['score']
        # if (not withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, position, keypointId)):
        notOverlappedKeypointScores += score

    return notOverlappedKeypointScores/len(instanceKeypoints)

def _sigmoid(z):
  return 1/(1 + np.exp(-z))

def decodeMultiplePoses(scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer, 
                        outputStride=16, maxPoseDetections=5, scoreThreshold=None, nmsRadius=None, kLocalMaximumRadius=1):
  return _decodeMultiplePoses(_sigmoid(np.asarray(scoresBuffer)), np.asarray(offsetsBuffer), np.asarray(displacementsFwdBuffer), np.asarray(displacementsBwdBuffer), 
                              outputStride, maxPoseDetections, scoreThreshold, nmsRadius, kLocalMaximumRadius)
def _decodeMultiplePoses(scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer, outputStride, 
                         maxPoseDetections, scoreThreshold, nmsRadius, kLocalMaximumRadius):
  if scoreThreshold == None: scoreThreshold = 0.5
  if nmsRadius == None: nmsRadius = 20
  poses = []
  queue = buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, scoresBuffer)
  squaredNmsRadius = nmsRadius * nmsRadius;
  while (len(poses) < maxPoseDetections) and len(queue):
    root = queue.pop()
    rootImageCoords = getImageCoords(root['part'], outputStride, offsetsBuffer)
    if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root['part']['id'])):
      continue
    keypoints = decodePose(root, rootImageCoords, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer, displacementsBwdBuffer)
    score = getInstanceScore(poses, squaredNmsRadius, keypoints)
    poses.append({ 'keypoints': keypoints, 'score': score })
  return poses

def draw_pose(pose, img, input_shape, threshold=0.5, marker_color='green', color='yellow', marker_size=5, thickness=2):
    # Resize and pad if necessary, the same way done for the inference
    img = ImageOps.pad(img, input_shape)
    draw = ImageDraw.Draw(img)

    for p1, p2 in poseChain:
        if (pose[p1]['score'] < threshold) or (pose[p2]['score'] < threshold): continue
        draw.line((pose[p1]['position']['x'], pose[p1]['position']['y'], pose[p2]['position']['x'], pose[p2]['position']['y']), fill=color, width=thickness)

    for label, keypoint in pose.items():
      if keypoint['score'] < threshold: continue
      draw.ellipse((int(keypoint['position']['x']-marker_size/2), 
                    int(keypoint['position']['y']-marker_size/2), 
                    int(keypoint['position']['x']+marker_size/2), 
                    int(keypoint['position']['y']+marker_size/2)), fill=marker_color)
      
    return img
