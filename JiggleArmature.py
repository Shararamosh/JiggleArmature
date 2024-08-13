# pylint: disable=invalid-name, unsubscriptable-object, too-many-instance-attributes
# pylint: disable=too-many-branches, too-many-statements, unsupported-assignment-operation
# pylint: disable=too-many-locals, too-many-nested-blocks
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
subject to the following conditions:
The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

This is a jiggle bone animation add-on for Blender. To enable jiggle physics first enable
"Jiggle Scene" in the Scene Properties and then enable "Jiggle Bone" on the Bones.


Based on the Position Based Dynamics paper by Müller et al.
http://matthias-mueller-fischer.ch/publications/posBasedDyn.pdf
"""
import math
from typing import Self
import bpy
from mathutils import Vector, Matrix, Quaternion
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, FloatVectorProperty
from bpy.utils import register_class, unregister_class

bl_info = {
    "name": "Jiggle Armature",
    "author": "Simón Flores, Shararamosh",
    "version": (2, 4, 0),
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
    def poll(cls, context: bpy.types.Context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw_header(self, context: bpy.types.Context):
        """
        Function for drawing header of Armature jiggle panel.
        """
        self.layout.prop(context.object.data.jiggle, "enabled", text="")

    def draw(self, context: bpy.types.Context):
        col = self.layout.column()
        self.layout.enabled = context.scene.jiggle.test_mode and context.object.data.jiggle.enabled
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

    def draw_header(self, context: bpy.types.Context):
        """
        Function for drawing header of Scene jiggle panel.
        """
        self.layout.prop(context.scene.jiggle, "test_mode", text="")

    def draw(self, context: bpy.types.Context):
        """
        Function for drawing body of Scene jiggle panel.
        """
        col = self.layout.column()
        self.layout.enabled = context.scene.jiggle.test_mode
        col.prop(context.scene.jiggle, "iterations", text="Iterations")
        col.operator("jiggle.bake", text="Bake Selected Jiggle Bones").a = False
        col.operator("jiggle.bake", text="Bake All Jiggle Bones").a = True


def fun_p(prop):
    """
    A function that changes jiggle bone property values for all selected jiggle bones.
    """
    in_f = False

    def f(_, context: bpy.types.Context):
        """
        Some kind of clever hack for Blender.
        """
        nonlocal in_f
        if in_f:
            return
        in_f = True
        o = context.object
        if o is None:
            in_f = False
            return
        b = context.bone
        arm = o.data
        for b2 in arm.bones:
            if b2.select:
                setattr(b2, prop, getattr(b, prop))
        in_f = False

    return f


def set_q(om, m):
    """
    A function that copies 4 indexed elements of mutable object into another one.
    """
    for i in range(4):
        om[i] = m[i]


def reset_jigglebone_state(context: bpy.types.Context):
    """
    Static function that resets current jiggle bone values.
    """
    scene = context.scene
    for o in scene.objects:
        if o.select_get() and o.type == 'ARMATURE':
            ow = o.matrix_world
            for b in o.pose.bones:
                if b.bone.select:
                    m = ow @ b.matrix
                    set_q(b.bone.jiggle_R, m.to_quaternion().normalized())
                    b.bone.jiggle_V = Vector((0, 0, 0))
                    b.bone.jiggle_P = m_pos(m) + m_axis(m, 1) * b.bone.length * 0.5
                    b.bone.jiggle_W = Vector((0, 0, 0))


class JARMOTReset(bpy.types.Operator):
    """
    Resets current jiggle bone values
    """
    bl_idname = "jiggle.reset"
    bl_label = "Reset Jiggle State"

    # noinspection PyMethodMayBeStatic
    def execute(self, context: bpy.types.Context):
        """
        Function that resets current jiggle bone values.
        """
        reset_jigglebone_state(context)
        return {'FINISHED'}


def set_jigglebone_rest_state(context: bpy.types.Context):
    """
    Static function that sets current bone values as jiggle bone rest pose.
    """
    scene = context.scene
    for o in scene.objects:
        if o.type == 'ARMATURE' and o.select_get():
            for b in o.pose.bones:
                if b.bone.select:
                    m = b.parent.matrix.inverted() @ b.matrix
                    set_q(b.bone.jiggle_rest, m.to_quaternion().normalized())
                    b.bone.jiggle_use_custom_rest = True


class JARMOTSetRest(bpy.types.Operator):
    """
    Sets current bone values as jiggle bone rest pose
    """
    bl_idname = "jiggle.set_rest"
    bl_label = "Set Rest Pose"

    # noinspection PyMethodMayBeStatic
    def execute(self, context: bpy.types.Context):
        """
        Function that sets current bone values as jiggle bone rest pose.
        """
        set_jigglebone_rest_state(context)
        return {'FINISHED'}


class JARMPTBone(bpy.types.Panel):
    """
    Class containing Bone jiggle panel.
    """
    bl_idname = "BONE_PT_jiggle_bone"
    bl_label = "Jiggle Bone"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "bone"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.object is None:
            return False
        if context.object.type != 'ARMATURE':
            return False
        return context.bone is not None

    def draw_header(self, context: bpy.types.Context):
        bon = context.bone
        self.layout.prop(bon, "jiggle_enabled", text="")

    def draw(self, context: bpy.types.Context):
        col = self.layout.column()
        armature = context.object.data
        self.layout.enabled = (context.scene.jiggle.test_mode and armature.jiggle.enabled
                               and context.bone.jiggle_enabled)
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


# adapted from https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/
# PositionBasedDynamics/PositionBasedRigidBodyDynamics.cpp
def compute_matrix_k(connector: Vector, inv_mass: float, x: Vector,
                     inertia_inverse_w: Matrix) -> Matrix:
    """
    Function that calculates K matrix.
    """
    k = Matrix().to_3x3()*0.0
    if inv_mass == 0.0:
        return k
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
    return k


class JB:
    """
    Class containing jiggle bone properties.
    """

    def __init__(self, b: bpy.types.PoseBone, m: Matrix, p: Self | None):
        self.m = m.copy()
        self.length = b.bone.length * m_axis(m, 0).length
        self.b = b
        self.parent = p
        self.rest = Matrix().to_3x3()*0.0
        self.rest_w = Matrix().to_3x3()*0.0
        self.w = 0.0
        self.k_c = 0.0
        self.c_q = None
        self.x = Vector((0, 0, 0))
        self.p = Matrix().to_3x3()*0.0
        self.r = Quaternion(Vector((0, 0, 0, 0)))
        self.q = Quaternion(Vector((0, 0, 0, 0)))
        self.i_i = Matrix.Identity(3)  # first naive approach
        self.i_iw = self.i_i
        self.b_len = 0.0
        self.rest_p = Vector((0, 0, 0))
        self.k = 0.0

    def compute_i(self):
        """
        No clue what this function does.
        """
        self.i_i = Matrix.Identity(3) * (self.w / (self.b_len * self.b_len) * 2.5)

    def update_iw(self):
        """
        No clue what this function does.
        """
        rot = self.q.to_matrix()
        self.i_iw = rot @ self.i_i
        self.i_iw @= rot.transposed()


def prop_b(ow: Matrix, b: bpy.types.PoseBone, jb_list: list[JB], p: JB | None):
    """
    Creates new JB object for current PoseBone and its children and appends them to list.
    """
    j = JB(b, ow @ b.matrix, p)
    jb_list.append(j)
    for c in b.children:
        prop_b(ow, c, jb_list, j)


def m_axis(m: Matrix, i: int):
    """
    Generates vector from one of matrix axes.
    """
    return Vector((m[0][i], m[1][i], m[2][i]))


def s_axis(m: Matrix, i, v: Vector):
    """
    Generates one of matrix axes from vector.
    """
    m[0][i] = v[0]
    m[1][i] = v[1]
    m[2][i] = v[2]


def m_pos(m: Matrix):
    """
    Generates vector from last matrix axis.
    """
    return m_axis(m, 3)


def ort(m: Matrix):
    """
    Generates orthogonal matrix from existing one.
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


def q_add(a: Quaternion, b: Quaternion):
    """
    Generates sum of 2 quaternions.
    """
    return Quaternion(Vector((a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])))


def q_add2(a: Quaternion, b: Quaternion):
    """
    Generates sum of 2 quaternions and applies values to first input Quaternion.
    """
    a.x += b.x
    a.y += b.y
    a.z += b.z
    a.w += b.w


def norm_r(m: Matrix):
    """
    Looks like normalizing matrix.
    """
    for i in range(3):
        s_axis(m, i, m_axis(m, i).normalized())


def loc_spring(jb: JB):
    """
    Looks like generating spring location.
    """
    w0 = jb.parent.w
    w1 = jb.w
    v0 = jb.rest_p
    p0 = jb.parent.p
    p1 = jb.p
    lf = jb.b_len * 0.5
    jb.update_iw()
    jb.parent.update_iw()
    connector0 = jb.parent.p + jb.parent.q @ v0
    connector1 = jb.p + jb.q @ Vector((0, -lf, 0))
    k1 = compute_matrix_k(connector0, w0, p0, jb.parent.i_iw)
    k2 = compute_matrix_k(connector1, w1, p1, jb.i_iw)
    k_inv = (k1 + k2).inverted()
    pt = k_inv @ (connector1 - connector0)
    if w0 != 0.0:
        r0 = connector0 - p0
        jb.parent.p += w0 * pt
        ot = (jb.parent.i_iw @ (r0.cross(pt)))
        ot_q = Quaternion(Vector((0, ot[0], ot[1], ot[2])))
        jb.parent.q = q_add(jb.parent.q, ot_q @ jb.parent.q * 0.5).normalized()
    if w1 != 0.0:
        r1 = connector1 - p1
        jb.p += -w1 * pt
        ot = (jb.i_iw @ (r1.cross(-pt)))
        ot_q = Quaternion(Vector((0, ot[0], ot[1], ot[2])))
        jb.q = q_add(jb.q, ot_q @ jb.q * 0.5).normalized()


# NOTE: the following gradient computation implementation was automatically generated, if possible,
# it should be changed for a clearer implementation
# Shara-add: Done.
def quat_spring_gradient2(q0: Quaternion, q1: Quaternion, r: Quaternion):
    """
    Returns the gradient of C = |Q0*r - Q1|^2 wrt Q0 and Q1.
    Heavily optimized by Shararamosh.
    """
    t1 = (((-(q0.x * q1.w) - (q0.y * q1.z)) + (q0.w * q1.x)) + (q0.z * q1.y)) - r.x
    t2 = (((-(q0.x * q1.y) - (q0.z * q1.w)) + (q0.w * q1.z)) + (q0.y * q1.x)) - r.z
    t3 = (((-(q0.y * q1.w) - (q0.z * q1.x)) + (q0.w * q1.y)) + (q0.x * q1.z)) - r.y
    t4 = ((((q0.w * q1.w) + (q0.x * q1.x)) + (q0.y * q1.y)) + (q0.z * q1.z)) - r.w
    tmp0 = math.sqrt((((math.pow(t1, 2) + math.pow(t2, 2)) + math.pow(t3, 2)) + math.pow(t4, 2)))
    tmp15 = 1.0 / tmp0 * q0.w * q0.w
    tmp16 = q1.w * q1.w
    tmp17 = 1.0 / tmp0 * q0.x * q0.x
    tmp18 = q1.x * q1.x
    tmp19 = 1.0 / tmp0 * q0.y * q0.y
    tmp20 = 1.0 / tmp0
    tmp21 = q1.y * q1.y
    tmp22 = tmp20 * q0.z * q0.z
    tmp23 = q1.z * q1.z
    tmp24 = tmp20 * q0.x
    tmp25 = tmp20 * q0.y
    tmp29 = tmp20 * q0.z
    tmp56 = tmp20 * q1.w
    tmp70 = tmp20 * q1.x
    tmp75 = tmp20 * q1.y
    tmp81 = tmp20 * q0.w
    tmp82 = tmp20 * q1.z
    temp_sum = tmp16 + tmp18 + tmp21 + tmp23
    d_q0x = tmp24 * temp_sum + tmp56 * r.x - tmp82 * r.y + tmp75 * r.z - tmp70 * r.w
    d_q0y = tmp25 * temp_sum + tmp82 * r.x + tmp56 * r.y - tmp70 * r.z - tmp75 * r.w
    d_q0z = tmp29 * temp_sum - tmp75 * r.x + tmp70 * r.y + tmp56 * r.z - tmp82 * r.w
    d_q0w = tmp81 * temp_sum - tmp70 * r.x - tmp75 * r.y - tmp82 * r.z - tmp56 * r.w
    temp_sum = tmp15 + tmp17 + tmp19 + tmp22
    d_q1x = q1.x * temp_sum - tmp81 * r.x + tmp29 * r.y - tmp25 * r.z - tmp24 * r.w
    d_q1y = q1.y * temp_sum - tmp29 * r.x - tmp81 * r.y + tmp24 * r.z - tmp25 * r.w
    d_q1z = q1.z * temp_sum + tmp25 * r.x - tmp24 * r.y - tmp81 * r.z - tmp29 * r.w
    d_q1w = q1.w * temp_sum + tmp24 * r.x + tmp25 * r.y + tmp29 * r.z - tmp81 * r.w
    return tmp0, d_q0x, d_q0y, d_q0z, d_q0w, d_q1x, d_q1y, d_q1z, d_q1w


def quat_spring(jb: JB, r: Quaternion = None, k: float | None = None):
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
    if (q0.inverted() @ q1).dot(r) < 0.0:
        r = -r
    c, d_q0x, d_q0y, d_q0z, d_q0w, d_q1x, d_q1y, d_q1z, d_q1w = quat_spring_gradient2(q0, q1, r)
    div = w0 * (math.pow(d_q0x, 2) + math.pow(d_q0y, 2) + math.pow(d_q0z, 2) + math.pow(d_q0w, 2))
    div += w1 * (math.pow(d_q1x, 2) + math.pow(d_q1y, 2) + math.pow(d_q1z, 2) + math.pow(d_q1w, 2))
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


def step(scene: bpy.types.Scene, dt: float):
    """
    One simulation step with delta time (in seconds).
    """
    dt /= scene.jiggle.sub_steps
    for o in scene.objects:
        if o.type != 'ARMATURE' or not o.data.jiggle.enabled:
            continue
        ow = o.matrix_world.copy()
        scale = m_axis(ow, 0).length
        frames = max(abs(int(dt * o.data.jiggle.fps)), 1)  # Amount of sub-frames for simulation.
        while frames > 0:
            frames -= 1
            bl = []
            for b in o.pose.bones:
                if b.parent is None:
                    prop_b(ow, b, bl, None)
            bl2 = []
            for wb in bl:
                b = wb.b
                wb.rest_w = b.bone.matrix_local.copy()
                s_axis(wb.rest_w, 3, m_axis(wb.rest_w, 3) * scale)
                temp = m_axis(wb.rest_w, 1) * b.bone.length * 0.5 * scale
                s_axis(wb.rest_w, 3, m_axis(wb.rest_w, 3) + temp)
            for wb in bl:
                b = wb.b
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
                    temp = (m_axis(wb.rest_w, 3) - m_axis(wb.rest_w, 1) * b.bone.length
                            * 0.5 * scale)
                    wb.rest_p = wb.parent.rest_w.inverted() @ temp
                    wb.b_len = b.bone.length * scale
                    wb.w = 1.0 / b.bone.jiggle_mass
                    wb.k = 1 - pow(1 - b.bone.jiggle_Ks, 1 / scene.jiggle.iterations)
                    b.bone.jiggle_V *= 1.0 - b.bone.jiggle_Kld
                    b.bone.jiggle_V += scene.gravity * dt
                    b.bone.jiggle_W *= 1.0 - b.bone.jiggle_Kd
                    qv = Quaternion(
                        Vector((0, b.bone.jiggle_W[0], b.bone.jiggle_W[1], b.bone.jiggle_W[2])))
                    wb.q = q_add(wb.q, qv @ wb.q * dt * 0.5).normalized()
                    wb.p = wb.x + b.bone.jiggle_V * dt
                    wb.compute_i()
                    # control object/bone constraint
                    if b.bone.jiggle_control_object in bpy.data.objects:
                        target_object = bpy.data.objects[b.bone.jiggle_control_object]
                        target_matrix = target_object.matrix_local
                        if (target_object.type == 'ARMATURE'
                                and b.bone.jiggle_control_bone in target_object.pose.bones):
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
                    set_q(b.bone.jiggle_R, m.to_quaternion().normalized())
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
                    quat_spring(wb, (b.bone.jiggle_rest if b.bone.jiggle_use_custom_rest
                                     else wb.rest.to_quaternion().normalized()))
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
def frame_change_post(scene: bpy.types.Scene):
    """
    Called when timeline frame was changed after all Blender's updates.
    """
    if scene.jiggle.test_mode:
        step(scene, scene.render.fps_base / scene.render.fps)


# noinspection PyUnresolvedReferences
def bake(context: bpy.types.Context, bake_all: bool):
    """
    Baking selected or all jiggle bones.
    """
    scene = context.scene
    if not scene.jiggle.test_mode:
        print("Can't bake Jiggle Bone transforms with Jiggle Scene disabled.")
        return
    print("Baking " + ("all" if bake_all else "selected") + " Jiggle Bones...")
    jiggle_armatures = []  # Keeping the list of Armatures that have at least one jiggle bone.
    for o in scene.objects:
        if o.type != 'ARMATURE' or not o.data.jiggle.enabled:
            continue
        if not bake_all and not o.select_get():
            continue
        ow = o.matrix_world
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
            set_q(b.bone.jiggle_R, m.to_quaternion().normalized())
            b.bone.jiggle_V = Vector((0, 0, 0))
            b.bone.jiggle_P = m_pos(m)
            b.bone.jiggle_W = Vector((0, 0, 0))
            b.bone.select = True
            has_any_jiggle_bone = True
        if has_any_jiggle_bone:
            jiggle_armatures.append((o, selected_bones))
    if len(jiggle_armatures) < 1:
        if bake_all:
            print("No Jiggle Bones were found in this Scene.")
        else:
            print("No Jiggle Bones are selected.")
        for jiggle_armature in jiggle_armatures:  # Restoring selected bones to each Armature.
            for b in jiggle_armature[0].pose.bones:
                b.bone.select = b in jiggle_armature[1]
        return
    current_frame = scene.frame_current
    previous_active = context.view_layer.objects.active
    scene.jiggle.test_mode = False
    for i in range(scene.frame_start, scene.frame_end + 1):
        print("Baking frame: ", str(i) + ".")
        scene.frame_set(i)
        step(scene, scene.render.fps_base / scene.render.fps)
        for jiggle_armature in jiggle_armatures:
            context.view_layer.objects.active = jiggle_armature[0]
            m = jiggle_armature[0].mode == 'POSE'
            if not m:
                bpy.ops.object.posemode_toggle()
            bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
            if bake_all or not m:
                bpy.ops.object.posemode_toggle()
    scene.frame_set(current_frame)
    scene.jiggle.test_mode = True
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
    bl_label = "Bake Jiggle Animation"

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
    bpy.types.Bone.jiggle_enabled = BoolProperty(default=False, update=fun_p("jiggle_enabled"))
    bpy.types.Bone.jiggle_Kld = FloatProperty(name="Linear damping", min=0.0, max=1.0, default=0.01,
                                              update=fun_p("jiggle_Kld"))
    bpy.types.Bone.jiggle_Kd = FloatProperty(name="Angular damping", min=0.0, max=1.0, default=0.01,
                                             update=fun_p("jiggle_Kd"))
    bpy.types.Bone.jiggle_Ks = FloatProperty(name="Stiffness", min=0.0, max=1.0, default=0.8,
                                             update=fun_p("jiggle_Ks"))
    bpy.types.Bone.jiggle_mass = FloatProperty(name="Mass", min=0.0001, default=1.0,
                                               update=fun_p("jiggle_mass"))
    bpy.types.Bone.jiggle_R = FloatVectorProperty(name="Rotation", size=4, subtype='QUATERNION')
    bpy.types.Bone.jiggle_W = FloatVectorProperty(size=3, subtype='XYZ')  # angular velocity
    bpy.types.Bone.jiggle_P = FloatVectorProperty(size=3, subtype='XYZ')
    bpy.types.Bone.jiggle_V = FloatVectorProperty(size=3, subtype='XYZ')  # linear velocity, ok?
    bpy.types.Bone.jiggle_use_custom_rest = BoolProperty(default=False, name="Use Custom Rest Pose",
                                                         update=fun_p("jiggle_use_custom_rest"))
    bpy.types.Bone.jiggle_rest = FloatVectorProperty(name="Rotation", size=4, subtype='QUATERNION')
    bpy.types.Bone.jiggle_control = FloatProperty(name="Control", min=0.0, max=1.0, default=1,
                                                  update=fun_p("jiggle_control"))
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
