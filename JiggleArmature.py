"""
Copyright (c) 2019 Simón Flores (https://github.com/cheece)

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so,
subject to the following conditions:The above copyright to_quaternion
notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

this a jiggle bone animation tool
to enable jiggle physics first enable "jiggle scene" in the scene properties and then enable jiggle
bone on the bones


based on the Position Based Dynamics paper by Müller et al.
http://matthias-mueller-fischer.ch/publications/posBasedDyn.pdf
"""
import math
import bpy
from mathutils import *
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, FloatVectorProperty
from bpy.utils import register_class, unregister_class

bl_info = {
    "name": "Jiggle Armature",
    "author": "Simón Flores, Shararamosh",
    "version": (2, 3, 0),
    "blender": (2, 80, 0),
    "description": "Jiggle Bone Animation Tool",
    "warning": "",
    "wiki_url": "",
    "category": "Animation",
}


class JiggleArmature(bpy.types.PropertyGroup):
    """
    Class containing Armature jiggle property.
    """
    enabled: BoolProperty(name="Enabled", default=True)
    fps: FloatProperty(name="Simulation FPS", default=24)


class JiggleScene(bpy.types.PropertyGroup):
    """
    Class containing Scene jiggle property.
    """
    test_mode: BoolProperty(default=False)
    sub_steps: IntProperty(name="Sub-steps", min=1, default=2)
    iterations: IntProperty(name="Iterations", min=1, default=4)


class JARMPTArmature(bpy.types.Panel):
    """
    Class containing Armature jiggle panel.
    """
    bl_idname = "ARMATURE_PT_jiggle"
    bl_label = "Jiggle Armature"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw_header(self, context):
        """
        Function for drawing header of Armature jiggle panel.
        """
        layout = self.layout
        layout.prop(context.object.data.jiggle, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        layout.enabled = context.scene.jiggle.test_mode and context.object.data.jiggle.enabled
        if not context.scene.jiggle.test_mode:
            col.label(text="Jiggle is disabled for this Scene, see Scene Properties.")
        col.prop(context.object.data.jiggle, "fps")


class JARMPTScene(bpy.types.Panel):
    """
    Class containing Scene jiggle panel.
    """
    bl_idname = "SCENE_PT_jiggle"
    bl_label = "Jiggle Scene"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        """
        Function for drawing header of Scene jiggle panel.
        """
        layout = self.layout
        layout.prop(context.scene.jiggle, "test_mode", text="")

    def draw(self, context):
        """
        Function for drawing body of Scene jiggle panel.
        """
        layout = self.layout
        col = layout.column()
        layout.enabled = context.scene.jiggle.test_mode
        col.prop(context.scene.jiggle, "iterations")
        col.operator("jiggle.bake", text="Bake Selected Jiggle Bones").a = False
        col.operator("jiggle.bake", text="Bake All Jiggle Bones").a = True


INOP = False


def funp(prop):
    """
    From my guess, it's a function that changes values of custom Blender properties.
    """

    def f(self, context):
        """
        Some kind of clever hack for Blender.
        """
        global INOP
        if INOP:
            return
        INOP = True
        b = context.bone
        o = context.object
        arm = o.data
        for b2 in arm.bones:
            if b2.select:
                setattr(b2, prop, getattr(b, prop))
        INOP = False

    return f


def setq(om, m):
    """
    Looks like a function that copies mutable 4 elements iterable to another one.
    As I can see it's used for copying quaternion values.
    """
    for i in range(4):
        om[i] = m[i]


def reset_jigglebone_state(context):
    """
    Static function that resets current jiggle bone values.
    """
    scene = context.scene
    for o in scene.objects:
        if o.select_get() and o.type == 'ARMATURE':
            # arm = o.data
            ow = o.matrix_world
            # scale = maxis(ow, 0).length
            # iow = ow.inverted()
            # i = 0
            for b in o.pose.bones:
                if b.bone.select:
                    m = ow @ b.matrix
                    setq(b.bone.jiggle_R, m.to_quaternion().normalized())
                    b.bone.jiggle_V = Vector((0, 0, 0))
                    b.bone.jiggle_P = m_pos(m) + m_axis(m, 1) * b.bone.length * 0.5
                    b.bone.jiggle_W = Vector((0, 0, 0))


class JARMOTReset(bpy.types.Operator):
    """
    Resets current jiggle bone values
    """
    bl_idname = "jiggle.reset"
    bl_label = "Reset State"

    def execute(self, context):
        """
        Function that resets current jiggle bone values.
        """
        reset_jigglebone_state(context)
        return {'FINISHED'}


def set_jigglebone_rest_state(context):
    """
    Static function that sets current bone values as jiggle bone rest pose.
    """
    scene = context.scene
    for o in scene.objects:
        if o.type == 'ARMATURE' and o.select_get():
            # arm = o.data
            # ow = o.matrix_world
            # scale = maxis(ow, 0).length
            # iow = ow.inverted()
            # i = 0
            for b in o.pose.bones:
                if b.bone.select:
                    m = b.parent.matrix.inverted() @ b.matrix  # ow*Sbp.wmat* Sb.rmat #im
                    setq(b.bone.jiggle_rest, m.to_quaternion().normalized())
                    b.bone.jiggle_use_custom_rest = True


class JARMOTSetRest(bpy.types.Operator):
    """
    Sets current bone values as jiggle bone rest pose
    """
    bl_idname = "jiggle.set_rest"
    bl_label = "Set Rest"

    def execute(self, context):
        """
        Function that sets current bone values as jiggle bone rest pose.
        """
        set_jigglebone_rest_state(context)
        return {'FINISHED'}


class JARMPTBone(bpy.types.Panel):
    """
    Class contaning Bone jiggle panel.
    """
    bl_idname = "BONE_PT_jiggle_bone"
    bl_label = "Jiggle Bone"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "bone"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        if context.object is None:
            return False
        if context.object.type != 'ARMATURE':
            return False
        return context.bone is not None

    def draw_header(self, context):
        layout = self.layout
        bon = context.bone
        layout.prop(bon, "jiggle_enabled", text="")

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        armature = context.object.data
        layout.enabled = (context.scene.jiggle.test_mode and armature.jiggle.enabled and
                          context.bone.jiggle_enabled)
        if not context.scene.jiggle.test_mode:
            col.label(text="Jiggle is disabled for this Scene, see Scene Properties.")
        elif not armature.jiggle.enabled:
            col.label(text="Jiggle is disabled for this Armature, see Armature Properties.")
        col.prop(context.bone, "jiggle_Ks")
        col.prop(context.bone, "jiggle_Kd")
        col.prop(context.bone, "jiggle_Kld")
        col.prop(context.bone, "jiggle_mass")
        col.prop_search(context.bone, "jiggle_control_object", bpy.data, "objects")
        if context.bone.jiggle_control_object in bpy.data.objects:
            o = bpy.data.objects[context.bone.jiggle_control_object]
            if o.type == 'ARMATURE':
                col.prop_search(context.bone, "jiggle_control_bone", o.data, "bones")
            col.prop(context.bone, "jiggle_control")
        col.operator("jiggle.reset")
        col.prop(context.bone, "jiggle_use_custom_rest")
        if context.bone.jiggle_use_custom_rest:
            col.prop(context.bone, "jiggle_rest")
            col.operator("jiggle.set_rest")
        if context.bone.parent is None:
            col.label(text="Warning: Jiggle bones without parent will fall.", icon='COLOR_RED')


def center_m(wb, l):
    """
    Some kind of centering.
    """
    ax = m_axis(wb, 1).normalized()
    wb[0][3] += ax[0] * l * 0.5
    wb[1][3] += ax[1] * l * 0.5
    wb[2][3] += ax[2] * l * 0.5


# adapted from https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/
# PositionBasedDynamics/PositionBasedRigidBodyDynamics.cpp
def compute_matrix_k(connector, inv_mass, x, inertia_inverse_w, k):
    """
    Function to calculate K matrix.
    """
    if inv_mass == 0.0:
        k.zero()
        return
    use_new_implementation = True
    v = connector - x
    a = v[0]
    b = v[1]
    c = v[2]
    if use_new_implementation:
        j11 = inertia_inverse_w[0][0]
        j12 = inertia_inverse_w[1][0]
        j13 = inertia_inverse_w[2][0]
        j22 = inertia_inverse_w[1][1]
        j23 = inertia_inverse_w[2][1]
        j33 = inertia_inverse_w[2][2]
        k[0][0] = c * c * j22 - b * c * (j23 + j23) + b * b * j33 + inv_mass
        k[1][0] = -(c * c * j12) + a * c * j23 + b * c * j13 - a * b * j33
        k[2][0] = b * c * j12 - a * c * j22 - b * b * j13 + a * b * j23
        k[0][1] = k[1][0]
        k[1][1] = c * c * j11 - a * c * (j13 + j13) + a * a * j33 + inv_mass
        k[2][1] = -(b * c * j11) + a * c * j12 + a * b * j13 - a * a * j23
        k[0][2] = k[2][0]
        k[1][2] = k[2][1]
        k[2][2] = b * b * j11 - a * b * (j12 + j12) + a * a * j22 + inv_mass
    else:
        j11 = inertia_inverse_w[0][0]
        j12 = inertia_inverse_w[0][1]
        j13 = inertia_inverse_w[0][2]
        j22 = inertia_inverse_w[1][1]
        j23 = inertia_inverse_w[1][2]
        j33 = inertia_inverse_w[2][2]
        k[0][0] = c * c * j22 - b * c * (j23 + j23) + b * b * j33 + inv_mass
        k[0][1] = -(c * c * j12) + a * c * j23 + b * c * j13 - a * b * j33
        k[0][2] = b * c * j12 - a * c * j22 - b * b * j13 + a * b * j23
        k[1][0] = k[0][1]
        k[1][1] = c * c * j11 - a * c * (j13 + j13) + a * a * j33 + inv_mass
        k[1][2] = -(b * c * j11) + a * c * j12 + a * b * j13 - a * a * j23
        k[2][0] = k[0][2]
        k[2][1] = k[1][2]
        k[2][2] = b * b * j11 - a * b * (j12 + j12) + a * a * j22 + inv_mass


class JB:
    """
    Class containing jiggle bone properties.
    """

    def __init__(self, b, m, p):
        self.m = m.copy()
        self.length = b.bone.length * m_axis(m, 0).length
        self.b = b
        self.parent = p
        self.rest = None
        self.rest_w = None
        self.w = 0
        self.k_c = 0
        self.c_q = None
        self.x = None
        self.p = None
        self.r = None
        self.q = None
        self.i_i = Matrix.Identity(3)  # first naive approach
        self.i_iw = self.i_i
        self.l = 0

    def compute_i(self):
        """
        This function may have error - l variable is not defined in class.
        """
        self.i_i = Matrix.Identity(3) * (self.w / (self.l * self.l) * 5.0 / 2.0)

    def update_iw(self):
        """
        No clue what this function does.
        """
        rot = self.q.to_matrix()
        self.i_iw = rot @ self.i_i @ rot.transposed()


def prop_b(ow, b, l, p):
    """
    No clue on what this function does.
    """
    j = JB(b, ow @ b.matrix, p)
    l.append(j)
    for c in b.children:
        prop_b(ow, c, l, j)


def m_axis(m, i):
    """
    Generates vector from one of matrix axes.
    """
    return Vector((m[0][i], m[1][i], m[2][i]))


def s_axis(m, i, v):
    """
    Generates one of matrix axes from vector.
    """
    m[0][i] = v[0]
    m[1][i] = v[1]
    m[2][i] = v[2]


def m_pos(m):
    """
    Generates vector from last matrix axis.
    """
    return m_axis(m, 3)
    # return Vector((m[0][3], m[1][3], m[2][3]))


def ort(m):
    """
    Generates orthogonal matrix.
    """
    a = m[0]
    b = m[1]
    c = m[2]
    a = a.normalized()
    b = (b - a * a.dot(b)).normalized()
    c = (c - a * a.dot(c) - b * b.dot(c)).normalized()
    m = Matrix.Identity(3)
    m[0] = a
    m[1] = b
    m[2] = c
    return m


def q_add(a, b):
    """
    Generates sum of 2 quaternions.
    """
    return Quaternion((a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))


def q_add2(a, b):
    """
    Generates sum of 2 quaternions.
    """
    a.x += b.x
    a.y += b.y
    a.z += b.z
    a.w += b.w


def norm_r(m):
    """
    Looks like normalizing matrix.
    """
    for i in range(3):
        s_axis(m, i, m_axis(m, i).normalized())


K1 = Matrix().to_3x3()
K2 = Matrix().to_3x3()


def loc_spring(jb):
    """
    Looks like generating spring location.
    """
    global K1
    global K2
    # q0 = jb.parent.Q
    # q1 = jb.Q
    w0 = jb.parent.w
    w1 = jb.w
    v0 = jb.rest_p
    p0 = jb.parent.p
    p1 = jb.p
    lf = jb.l * 0.5
    jb.update_iw()
    jb.parent.update_iw()
    connector0 = jb.parent.p + jb.parent.q @ v0
    connector1 = jb.p + jb.q @ Vector((0, -lf, 0))
    compute_matrix_k(connector0, w0, p0, jb.parent.i_iw, K1)
    compute_matrix_k(connector1, w1, p1, jb.i_iw, K2)
    k_inv = (K1 + K2).inverted()
    pt = k_inv @ (connector1 - connector0)
    if w0 != 0.0:
        r0 = connector0 - p0
        jb.parent.p += w0 * pt
        ot = (jb.parent.i_iw @ (r0.cross(pt)))
        ot_q = Quaternion()
        ot_q.x = ot[0]
        ot_q.y = ot[1]
        ot_q.z = ot[2]
        ot_q.w = 0
        jb.parent.q = q_add(jb.parent.q, ot_q @ jb.parent.q * 0.5).normalized()
    if w1 != 0.0:
        r1 = connector1 - p1
        jb.p += -w1 * pt
        ot = (jb.i_iw @ (r1.cross(-pt)))
        ot_q = Quaternion()
        ot_q.x = ot[0]
        ot_q.y = ot[1]
        ot_q.z = ot[2]
        ot_q.w = 0
        jb.q = q_add(jb.q, ot_q @ jb.q * 0.5).normalized()


# NOTE: the following gradient computation implementation was automatically generated, if possible,
# it should be changed for a clearer implementation
def quat_spring_gradient2(q0, q1, r):
    """
    Returns the gradient of C = |Q0*r - Q1|^2 wrt Q0 and Q1.
    """
    q0x = q0.x
    q0y = q0.y
    q0z = q0.z
    q0w = q0.w
    q1x = q1.x
    q1y = q1.y
    q1z = q1.z
    q1w = q1.w
    rx = r.x
    ry = r.y
    rz = r.z
    rw = r.w
    tmp0 = math.sqrt(((((((((-(q0x * q1w) - (q0y * q1z)) + (q0w * q1x)) + (q0z * q1y)) - rx) * (
            (((-(q0x * q1w) - (q0y * q1z)) + (q0w * q1x)) + (q0z * q1y)) - rx)) + (((((-(
            q0x * q1y) - (q0z * q1w)) + (q0w * q1z)) + (q0y * q1x)) - rz) * ((((-(q0x * q1y) - (
            q0z * q1w)) + (q0w * q1z)) + (q0y * q1x)) - rz))) + (((((-(q0y * q1w) - (
            q0z * q1x)) + (q0w * q1y)) + (q0x * q1z)) - ry) * ((((-(q0y * q1w) - (
            q0z * q1x)) + (q0w * q1y)) + (q0x * q1z)) - ry))) + ((((((q0w * q1w) + (
            q0x * q1x)) + (q0y * q1y)) + (q0z * q1z)) - rw) * (((((q0w * q1w) + (q0x * q1x)) + (
            q0y * q1y)) + (q0z * q1z)) - rw))))
    tmp1 = 1.0 / tmp0 * q0w * q0y
    tmp2 = 1.0 / tmp0 * q0w * q1x
    tmp3 = 1.0 / tmp0 * q0w * q0x
    tmp4 = 1.0 / tmp0 * q0x * q1w
    tmp5 = 1.0 / tmp0 * q0w * q1w
    tmp6 = 1.0 / tmp0 * q0y * q1w
    tmp7 = 1.0 / tmp0 * q0w * q0z
    tmp8 = 1.0 / tmp0 * q0x * q1x
    tmp9 = 1.0 / tmp0 * q0y * q1x
    tmp10 = 1.0 / tmp0 * q0x * q0y
    tmp11 = 1.0 / tmp0 * q0x * q0z
    tmp12 = 1.0 / tmp0 * q0z * q1w
    tmp13 = 1.0 / tmp0 * q0z * q1x
    tmp14 = 1.0 / tmp0 * q0y * q0z
    tmp15 = 1.0 / tmp0 * q0w * q0w
    tmp16 = q1w * q1w
    tmp17 = 1.0 / tmp0 * q0x * q0x
    tmp18 = q1x * q1x
    tmp19 = 1.0 / tmp0 * q0y * q0y
    tmp20 = 1.0 / tmp0
    tmp21 = q1y * q1y
    tmp22 = tmp20 * q0z * q0z
    tmp23 = q1z * q1z
    tmp24 = tmp20 * q0x
    tmp25 = tmp20 * q0y
    tmp26 = tmp4 * q1x
    tmp27 = tmp24 * q1y * q1z
    tmp28 = tmp3 * q1y
    tmp29 = tmp20 * q0z
    tmp30 = tmp3 * q1z
    tmp31 = tmp3 * q1w
    tmp32 = tmp5 * q1y
    tmp33 = tmp5 * q1z
    tmp34 = tmp1 * q1z
    tmp35 = tmp5 * q1x
    tmp36 = tmp1 * q1x
    tmp37 = tmp1 * q1w
    tmp38 = tmp6 * q1y
    tmp39 = tmp7 * q1y
    tmp40 = tmp2 * q1z
    tmp41 = tmp7 * q1x
    tmp42 = tmp9 * q1z
    tmp43 = tmp2 * q1y
    tmp44 = tmp3 * q1x
    tmp45 = tmp7 * q1w
    tmp46 = tmp20 * q0w * q1y * q1z
    tmp47 = tmp10 * q1x
    tmp48 = tmp4 * q1z
    tmp49 = tmp10 * q1y
    tmp50 = tmp10 * q1w
    tmp51 = tmp6 * q1z
    tmp52 = tmp4 * q1y
    tmp53 = tmp1 * q1y
    tmp54 = tmp12 * q1z
    # tmp55 = -q0x * q1w - q0y * q1z + q0w * q1x + q0z * q1y - rx
    tmp56 = tmp20 * q1w
    tmp57 = tmp11 * q1z
    tmp58 = tmp9 * q1y
    tmp59 = tmp7 * q1z
    tmp60 = tmp11 * q1x
    tmp61 = tmp8 * q1y
    tmp62 = tmp13 * q1y
    tmp63 = tmp11 * q1w
    tmp64 = tmp8 * q1z
    # tmp65 = -q0x * q1y - q0z * q1w + q0w * q1z + q0y * q1x - rz
    tmp66 = tmp14 * q1y
    tmp67 = tmp14 * q1w
    tmp68 = tmp12 * q1y
    tmp69 = tmp14 * q1z
    tmp70 = tmp20 * q1x
    # tmp71 = -q0y * q1w - q0z * q1x + q0w * q1y + q0x * q1z - ry
    tmp72 = tmp6 * q1x
    tmp73 = tmp10 * q1z
    tmp74 = tmp12 * q1x
    tmp75 = tmp20 * q1y
    # tmp76 = q0w * q1w + q0x * q1x + q0y * q1y + q0z * q1z - rw
    tmp77 = tmp29 * q1y * q1z
    tmp78 = tmp25 * q1y * q1z
    tmp79 = tmp13 * q1z
    tmp80 = tmp11 * q1y
    tmp81 = tmp20 * q0w
    tmp82 = tmp20 * q1z
    tmp83 = tmp14 * q1x
    c = tmp0
    d_q0x = (tmp35 + tmp46 + tmp51 + tmp58 + tmp68 + tmp79 + tmp24 * tmp16 + tmp24 * tmp18 + tmp24
             * tmp21 + tmp24 * tmp23 + tmp56 * rx + tmp75 * rz - tmp35 - tmp46 - tmp51 - tmp58
             - tmp68 - tmp79 - tmp70 * rw - tmp82 * ry)
    d_q0y = (tmp32 + tmp40 + tmp48 + tmp61 + tmp74 + tmp77 + tmp25 * tmp16 + tmp25 * tmp18 + tmp25
             * tmp21 + tmp25 * tmp23 + tmp56 * ry + tmp82 * rx - tmp32 - tmp40 - tmp48 - tmp61
             - tmp74 - tmp77 - tmp70 * rz - tmp75 * rw)
    d_q0z = (tmp33 + tmp43 + tmp52 + tmp64 + tmp72 + tmp78 + tmp29 * tmp16 + tmp29 * tmp18
             + tmp29
             * tmp21 + tmp29 * tmp23 + tmp56 * rz + tmp70 * ry - tmp33 - tmp43 - tmp52
             - tmp64 - tmp72 - tmp78 - tmp75 * rx - tmp82 * rw)
    d_q0w = (tmp26 + tmp27 + tmp38 + tmp42 + tmp54 + tmp62 + tmp81 * tmp16 + tmp81 * tmp18 + tmp81
             * tmp21 + tmp81 * tmp23 - tmp26 - tmp27 - tmp38 - tmp42 - tmp54 - tmp62 - tmp56 * rw
             - tmp70 * rx - tmp75 * ry - tmp82 * rz)
    d_q1x = (tmp31 + tmp34 + tmp39 + tmp49 + tmp57 + tmp67 + tmp15 * q1x + tmp17 * q1x + tmp19 * q1x
             + tmp22 * q1x + tmp29 * ry - tmp31 - tmp34 - tmp39 - tmp49 - tmp57 - tmp67 - tmp81 * rx
             - tmp24 * rw - tmp25 * rz)
    d_q1y = (tmp30 + tmp37 + tmp41 + tmp47 + tmp63 + tmp69 + tmp15 * q1y + tmp17 * q1y + tmp19 * q1y
             + tmp22 * q1y + tmp24 * rz - tmp30 - tmp37 - tmp41 - tmp47 - tmp63 - tmp69 - tmp81 * ry
             - tmp25 * rw - tmp29 * rx)
    d_q1z = (tmp28 + tmp36 + tmp45 + tmp50 + tmp60 + tmp66 + tmp15 * q1z + tmp17 * q1z + tmp19 * q1z
             + tmp22 * q1z + tmp25 * rx - tmp28 - tmp36 - tmp45 - tmp50 - tmp60 - tmp66 - tmp81 * rz
             - tmp24 * ry - tmp29 * rw)
    d_q1w = (tmp44 + tmp53 + tmp59 + tmp73 + tmp80 + tmp83 + tmp15 * q1w + tmp17 * q1w + tmp19 * q1w
             + tmp22 * q1w + tmp24 * rx + tmp25 * ry + tmp29 * rz - tmp44 - tmp53 - tmp59 - tmp73 -
             tmp80 - tmp83 - tmp81 * rw)
    return c, d_q0x, d_q0y, d_q0z, d_q0w, d_q1x, d_q1y, d_q1z, d_q1w


def quat_spring(jb, r=None, k=None):
    """
    From my guess calculates spring quaternion.
    """
    q0 = jb.parent.q
    q1 = jb.q
    w0 = jb.parent.w
    w1 = jb.w
    if r is None:
        r = jb.rest.to_quaternion()
    if k is None:
        k = jb.k
    ra = q0.inverted() @ q1
    if ra.dot(r) < 0:
        r = -r
    c, d_q0x, d_q0y, d_q0z, d_q0w, d_q1x, d_q1y, d_q1z, d_q1w = quat_spring_gradient2(q0, q1, r)
    div = d_q0x * d_q0x * w0 + d_q0y * d_q0y * w0 + d_q0z * d_q0z * w0 + d_q0w * d_q0w * w0 + d_q1x * d_q1x * w1 + d_q1y * d_q1y * w1 + d_q1z * d_q1z * w1 + d_q1w * d_q1w * w1
    if div > 1e-8:
        s = -c / div
        if w0 > 0.0:
            q0.x += d_q0x * s * w0 * k
            q0.y += d_q0y * s * w0 * k
            q0.z += d_q0z * s * w0 * k
            q0.w += d_q0w * s * w0 * k
            jb.parent.q = q0.normalized()
        q1.x += d_q1x * s * w1 * k
        q1.y += d_q1y * s * w1 * k
        q1.z += d_q1z * s * w1 * k
        q1.w += d_q1w * s * w1 * k
        jb.q = q1.normalized()


def step(scene, dt):
    """
    One simulation step with delta time (in seconds).
    """
    dt /= scene.jiggle.sub_steps
    for o in scene.objects:
        if o.type == 'ARMATURE' and o.data.jiggle.enabled:
            arm = o.data
            ow = o.matrix_world.copy()
            scale = m_axis(ow, 0).length
            # iow = ow.inverted()
            # iow3 = ow.to_3x3().inverted()
            # i = 0
            frames = max(abs(int(dt * arm.jiggle.fps)), 1)  # Amount of frames simulation
            # should go for.
            while frames > 0:
                frames -= 1
                bl = []
                # wt = []
                for b in o.pose.bones:
                    if b.parent is None:
                        prop_b(ow, b, bl, None)
                # hooks = []
                bl2 = []
                for wb in bl:
                    b = wb.b
                    wb.rest_w = b.bone.matrix_local.copy()
                    s_axis(wb.rest_w, 3, m_axis(wb.rest_w, 3) * scale)
                    s_axis(wb.rest_w, 3,
                           m_axis(wb.rest_w, 3) + m_axis(wb.rest_w,
                                                         1) * b.bone.length * 0.5 * scale)
                for wb in bl:
                    b = wb.b
                    # crest = b
                    wb.restW = b.bone.matrix_local.copy() * scale
                    s_axis(wb.restW, 3, m_axis(wb.restW, 3) * scale)
                    m = wb.m
                    if b.bone.jiggle_enabled:
                        wb.x = wb.p = b.bone.jiggle_P
                        wb.r = wb.q = b.bone.jiggle_R
                        wb.rest = wb.rest_w
                        if b.parent is not None:
                            wb.rest = wb.parent.rest_w.inverted() @ wb.rest_w
                        wb.rest_base = b.bone.matrix_local
                        if b.parent is not None:
                            wb.rest_base = b.parent.bone.matrix_local.inverted() @ wb.rest_base
                        wb.rest_p = wb.parent.rest_w.inverted() @ (
                                m_axis(wb.rest_w, 3) - m_axis(wb.rest_w,
                                                              1) * b.bone.length * 0.5 * scale)  # mpos(wb.rest)
                        wb.l = b.bone.length * scale
                        wb.w = 1.0 / b.bone.jiggle_mass
                        wb.k = 1 - pow(1 - b.bone.jiggle_Ks, 1 / scene.jiggle.iterations)
                        b.bone.jiggle_V *= 1.0 - b.bone.jiggle_Kld
                        b.bone.jiggle_V += scene.gravity * dt
                        b.bone.jiggle_W *= 1.0 - b.bone.jiggle_Kd
                        qv = Quaternion()
                        qv.x = b.bone.jiggle_W[0]
                        qv.y = b.bone.jiggle_W[1]
                        qv.z = b.bone.jiggle_W[2]
                        qv.w = 0
                        wb.q = q_add(wb.q, qv @ wb.q * dt * 0.5).normalized()
                        wb.p = wb.x + b.bone.jiggle_V * dt
                        wb.compute_i()
                        # control object/bone constraint
                        if b.bone.jiggle_control_object in bpy.data.objects:
                            target_object = bpy.data.objects[b.bone.jiggle_control_object]
                            target_matrix = target_object.matrix_local
                            if target_object.type == 'ARMATURE' and b.bone.jiggle_control_bone in target_object.pose.bones:
                                cb = target_object.pose.bones[b.bone.jiggle_control_bone]
                                target_matrix = cb.matrix
                                if cb.parent is not None:
                                    target_matrix = cb.parent.matrix.inverted() @ target_matrix
                            wb.c_q = target_matrix.to_quaternion().normalized()
                            wb.k_c = 1 - pow(1 - b.bone.jiggle_control,
                                             1.0 / scene.jiggle.iterations)
                        bl2.append(wb)
                    else:
                        wb.w = 0
                        wb.x = wb.p = m_pos(m) + m_axis(m, 1) * b.bone.length * 0.5
                        wb.r = wb.q = m.to_quaternion().normalized()
                        m = ow @ b.matrix
                        setq(b.bone.jiggle_R, m.to_quaternion().normalized())
                        b.bone.jiggle_V = Vector((0, 0, 0))
                        b.bone.jiggle_P = m_pos(m) + m_axis(m, 1) * b.bone.length * 0.5
                        b.bone.jiggle_W = Vector((0, 0, 0))
                for i in range(scene.jiggle.iterations):
                    # parent constraint
                    for wb in bl2:
                        b = wb.b
                        if b.parent is None:
                            continue
                        loc_spring(wb)
                        # spring constraint
                    for wb in bl2:
                        b = wb.b
                        if b.parent is None:
                            continue
                        quat_spring(wb,
                                    b.bone.jiggle_rest if b.bone.jiggle_use_custom_rest else wb.rest.to_quaternion().normalized())
                        if wb.c_q is not None:
                            quat_spring(wb, wb.c_q, wb.k_c)
                for wb in bl2:
                    b = wb.b
                    wb.q = wb.q.normalized()
                    m = wb.q.to_matrix()
                    for i in range(3):
                        for j in range(3):
                            wb.m[i][j] = m[i][j] * scale
                    wb.m[3][3] = 1
                    b.bone.jiggle_V = (wb.p - wb.x) / dt
                    b.bone.jiggle_P = wb.p.copy()
                    qv = wb.q @ b.bone.jiggle_R.conjugated()
                    b.bone.jiggle_W = Vector((qv.x, qv.y, qv.z)) * (2 / dt)
                    b.bone.jiggle_R = wb.q
                    cp = b.bone.jiggle_P - m_axis(wb.m, 1) * b.bone.length * 0.5
                    wb.m[0][3] = cp[0]
                    wb.m[1][3] = cp[1]
                    wb.m[2][3] = cp[2]
                for wb in bl2:
                    b = wb.b
                    p_m = ow
                    if b.parent is not None:
                        p_m = wb.parent.m
                    mb = (p_m @ wb.rest_base).inverted() @ wb.m
                    b.matrix_basis = mb


@persistent
def frame_change_post(scene):
    """
    Called when timeline frame was changed after all Blender's updates.
    """
    step(scene, scene.render.fps_base / scene.render.fps)


def bake(context, bake_all: bool):
    """
    Baking selected or all jiggle bones.
    """
    scene = context.scene
    if not scene.jiggle.test_mode:
        print("Can't bake jiggle bone transforms with scene jiggle disabled.")
        return
    print("Baking " + ("all" if bake_all else "selected") + " jiggle bones...")
    scene.frame_set(scene.frame_start)
    jiggle_armatures = []  # Keeping the list of Armatures that have at least one jiggle bone.
    for o in scene.objects:
        if o.type != 'ARMATURE' or not o.data.jiggle.enabled:
            continue
        if not bake_all and not o.select_get():
            continue
        # arm = o.data
        ow = o.matrix_world
        # scale = maxis(ow, 0).length
        # iow = ow.inverted()
        i = 0
        has_any_jiggle_bone = False
        selected_bones = []
        for b in o.pose.bones:
            if b.bone.select:
                selected_bones.append(b)
            if not b.bone.jiggle_enabled:
                b.bone.select = False
                continue
            if not bake_all and not b.bone.select:
                continue
            m = ow @ b.matrix
            setq(b.bone.jiggle_R, m.to_quaternion().normalized())
            b.bone.jiggle_V = Vector((0, 0, 0))
            b.bone.jiggle_P = m_pos(m)
            b.bone.jiggle_W = Vector((0, 0, 0))
            b.bone.select = True
            has_any_jiggle_bone = True
        if has_any_jiggle_bone:
            jiggle_armatures.append((o, selected_bones))
    if len(jiggle_armatures) < 1:
        if bake_all:
            print("No jiggle bones were found in this Scene.")
        else:
            print("No jiggle bones are selected.")
        for jiggle_armature in jiggle_armatures:  # Restoring selected bones to each Armature.
            for b in jiggle_armature[0].pose.bones:
                b.bone.select = b in jiggle_armature[1]
        return
    previous_active = context.view_layer.objects.active
    for i in range(scene.frame_start, scene.frame_end + 1):
        print("Baking frame: ", str(i) + ".")
        scene.frame_set(i)
        for jiggle_armature in jiggle_armatures:
            context.view_layer.objects.active = jiggle_armature[0]
            m = jiggle_armature[0].mode == 'POSE'
            if not m:
                bpy.ops.object.posemode_toggle()
            bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
            if bake_all or not m:
                bpy.ops.object.posemode_toggle()
    context.view_layer.objects.active = previous_active
    for jiggle_armature in jiggle_armatures:
        for b in jiggle_armature[0].pose.bones:
            b.bone.select = b in jiggle_armature[1]  # Restoring selected bones to each
            # baked Armature.
        for b in jiggle_armature[1]:  # Disabling jiggle on all baked jiggle bones.
            b.bone.jiggle_enabled = False
        if bake_all:  # Disabling jiggle on all baked Armatures if baked all.
            jiggle_armature[0].data.jiggle.enabled = False


class JARMOTBake(bpy.types.Operator):
    """
    Bake jiggle bone values to current action
    """
    a: BoolProperty()
    bl_idname = "jiggle.bake"
    bl_label = "Bake Animation"

    def execute(self, context):
        bake(context, self.a)
        return {'FINISHED'}


classes = (JARMPTArmature, JiggleScene, JARMPTScene, JiggleArmature, JARMOTBake, JARMOTSetRest,
           JARMOTReset, JARMPTBone)


def register():
    """
    Function for registering addon.
    """
    for cls in classes:
        register_class(cls)
    bpy.app.handlers.frame_change_post.append(frame_change_post)
    bpy.types.Scene.jiggle = bpy.props.PointerProperty(type=JiggleScene)
    bpy.types.Armature.jiggle = bpy.props.PointerProperty(type=JiggleArmature,
                                                          options={'ANIMATABLE'})
    bpy.types.Bone.jiggle_enabled = BoolProperty(default=False, update=funp("jiggle_enabled"))
    bpy.types.Bone.jiggle_Kld = FloatProperty(name="Linear damping", min=0.0, max=1.0, default=0.01,
                                              update=funp("jiggle_Kld"))
    bpy.types.Bone.jiggle_Kd = FloatProperty(name="Angular damping", min=0.0, max=1.0, default=0.01,
                                             update=funp("jiggle_Kd"))
    bpy.types.Bone.jiggle_Ks = FloatProperty(name="Stiffness", min=0.0, max=1.0, default=0.8,
                                             update=funp("jiggle_Ks"))
    bpy.types.Bone.jiggle_mass = FloatProperty(name="Mass", min=0.0001, default=1.0,
                                               update=funp("jiggle_mass"))
    bpy.types.Bone.jiggle_R = FloatVectorProperty(name="Rotation", size=4, subtype='QUATERNION')
    bpy.types.Bone.jiggle_W = FloatVectorProperty(size=3, subtype='XYZ')  # angular velocity
    bpy.types.Bone.jiggle_P = FloatVectorProperty(size=3, subtype='XYZ')
    bpy.types.Bone.jiggle_V = FloatVectorProperty(size=3, subtype='XYZ')  # linear velocity, ok?
    bpy.types.Bone.jiggle_use_custom_rest = BoolProperty(default=False, name="Use Custom Rest Pose",
                                                         update=funp("jiggle_use_custom_rest"))
    bpy.types.Bone.jiggle_rest = FloatVectorProperty(name="Rotation", size=4, subtype='QUATERNION')
    bpy.types.Bone.jiggle_control = FloatProperty(name="Control", min=0.0, max=1.0, default=1,
                                                  update=funp("jiggle_control"))
    bpy.types.Bone.jiggle_control_object = bpy.props.StringProperty(name="Control Object")
    bpy.types.Bone.jiggle_control_bone = bpy.props.StringProperty(name="Control Bone")


def unregister():
    """
    Function for unregistering addon.
    """
    for cls in reversed(classes):
        unregister_class(cls)
    bpy.app.handlers.frame_change_post.remove(frame_change_post)


if __name__ == '__main__':
    register()
