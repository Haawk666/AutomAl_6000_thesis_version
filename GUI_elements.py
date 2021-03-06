# By Haakon Tvedt @ NTNU
"""Module container for high-level custom GUI-elements"""

# Program imports:
import GUI_custom_components
import GUI_settings
import GUI_tooltips
import GUI
import utils
import data_module
# External imports:
import configparser
import copy
from PyQt5 import QtWidgets, QtGui, QtCore
import pathlib
import os
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ----------
# Custom QGraphicsScene and QGraphicsView classes:
# ----------


class RawImage(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **raw image**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        self.scale_bar = None
        if self.ui_obj.project_instance is not None:
            self.scale_bar = GUI_custom_components.ScaleBar(length=2, scale=self.ui_obj.project_instance.scale, r=self.ui_obj.project_instance.r, height=self.ui_obj.project_instance.im_height)
            self.addItem(self.scale_bar)
            if self.ui_obj.control_window.chb_scalebar.isChecked():
                self.scale_bar.show()
            else:
                self.scale_bar.hide()


class StaticImage(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **raw image**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)


class AtomicPositions(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic positions**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.interactive_position_objects = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)

    def re_draw(self):
        """Redraw contents."""
        self.interactive_position_objects = []
        if self.ui_obj.project_instance is not None:
            for vertex in self.ui_obj.project_instance.graph.vertices:
                self.interactive_position_objects.append(GUI_custom_components.InteractivePosColumn(
                    ui_obj=self.ui_obj,
                    vertex=vertex,
                    r=vertex.r
                ))
                self.addItem(self.interactive_position_objects[-1])
                if not self.ui_obj.control_window.chb_toggle_positions.isChecked():
                    self.interactive_position_objects[-1].hide()
                else:
                    self.interactive_position_objects[-1].show()
                if vertex.void:
                    self.interactive_position_objects[-1].hide()


class OverlayComposition(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **overlay composition**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.interactive_overlay_objects = []
        self.background_image = background
        self.pixmap = None
        self.categories = ['si', 'cu', 'al', 'mg', 'un']
        if background is not None:
            self.pixmap = self.addPixmap(self.background_image)
            if self.ui_obj.control_window.chb_raw_image.isChecked():
                self.pixmap.show()
            else:
                self.pixmap.hide()
        if self.ui_obj.control_window.chb_black_background.isChecked():
            self.setBackgroundBrush(GUI_settings.brush_black)
        else:
            if GUI_settings.theme == 'dark':
                self.setBackgroundBrush(GUI_settings.background_brush)
            else:
                self.setBackgroundBrush(GUI_settings.brush_white)
        self.scale_bar = None
        self.parser = None
        self.zeta_1_brushes = []
        self.zeta_0_brushes = []
        self.pens = []
        if self.ui_obj.project_instance is not None:
            self.scale_bar = GUI_custom_components.ScaleBar(
                length=2,
                scale=self.ui_obj.project_instance.scale,
                r=self.ui_obj.project_instance.r,
                height=self.ui_obj.project_instance.im_height
            )
            self.addItem(self.scale_bar)
            self.legend = GUI_custom_components.Legend(
                ui_obj=self.ui_obj
            )
            self.addItem(self.legend)
            self.scale_bar.setZValue(2)
            if self.ui_obj.control_window.chb_scalebar.isChecked():
                self.scale_bar.show()
            else:
                self.scale_bar.hide()
            self.parser = configparser.ConfigParser()
            self.parser.read('config.ini')
            self.radii = []
            self.pens = []
            self.zeta_1_brushes = []
            self.zeta_0_brushes = []
            for species in self.categories:
                self.radii.append(self.parser.getfloat('overlay_radii', '{}_outer'.format(species)) / self.ui_obj.project_instance.scale)
                color_1 = QtGui.QColor(
                    self.parser.getint('colors', '{}_primary_r'.format(species)),
                    self.parser.getint('colors', '{}_primary_g'.format(species)),
                    self.parser.getint('colors', '{}_primary_b'.format(species))
                )
                color_2 = QtGui.QColor(
                    self.parser.getint('colors', '{}_secondary_r'.format(species)),
                    self.parser.getint('colors', '{}_secondary_g'.format(species)),
                    self.parser.getint('colors', '{}_secondary_b'.format(species))
                )
                pen = QtGui.QPen(color_1)
                r_factor = self.parser.getfloat('overlay_radii', '{}_inner'.format(species)) / self.ui_obj.project_instance.scale
                pen.setWidth(r_factor)
                self.pens.append(pen)
                brush = QtGui.QBrush(color_1)
                self.zeta_0_brushes.append(brush)
                brush = QtGui.QBrush(color_2)
                self.zeta_1_brushes.append(brush)

    def re_draw(self):
        """Redraw contents."""
        self.interactive_overlay_objects = []
        if self.ui_obj.project_instance is not None:
            for vertex in self.ui_obj.project_instance.graph.vertices:
                species = vertex.atomic_species.lower()
                if vertex.zeta == 0:
                    brush = self.zeta_0_brushes[self.categories.index(species)]
                else:
                    brush = self.zeta_1_brushes[self.categories.index(species)]
                self.interactive_overlay_objects.append(GUI_custom_components.InteractiveOverlayColumn(
                    ui_obj=self.ui_obj,
                    vertex=vertex,
                    r=self.radii[self.categories.index(species)],
                    pen=self.pens[self.categories.index(species)],
                    brush=brush
                ))
                self.addItem(self.interactive_overlay_objects[-1])
                if vertex.show_in_overlay:
                    self.interactive_overlay_objects[-1].show()
                else:
                    self.interactive_overlay_objects[-1].hide()
                if vertex.void:
                    self.interactive_overlay_objects[-1].hide()

    def re_draw_vertex(self, i):
        self.removeItem(self.interactive_overlay_objects[i])
        vertex = self.ui_obj.project_instance.graph.vertices[i]
        species = vertex.atomic_species.lower()
        if vertex.zeta == 0:
            brush = self.zeta_0_brushes[self.categories.index(species)]
        else:
            brush = self.zeta_1_brushes[self.categories.index(species)]
        self.interactive_overlay_objects[i] = GUI_custom_components.InteractiveOverlayColumn(
            ui_obj=self.ui_obj,
            vertex=vertex,
            r=self.radii[self.categories.index(species)],
            pen=self.pens[self.categories.index(species)],
            brush=brush
        )
        self.addItem(self.interactive_overlay_objects[i])
        if vertex.show_in_overlay:
            self.interactive_overlay_objects[-1].show()
        else:
            self.interactive_overlay_objects[-1].hide()
        if vertex.void:
            self.interactive_overlay_objects[-1].hide()


class AtomicGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, scale_factor=1, mode='district'):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.scale_factor = scale_factor
        self.mode = mode
        self.interactive_vertex_objects = []
        self.arcs = []
        self.mesh_details = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        if GUI_settings.theme == 'dark':
            self.setBackgroundBrush(GUI_settings.background_brush)
        self.re_draw()

    def perturb_edge(self, i, j, k, permute_data=True, center_view=False):
        """Finds the edge from i to j, and makes it point from i to k."""
        if permute_data:
            if self.mode == 'district':
                self.ui_obj.project_instance.graph.permute_j_k(i, j, k)
            elif self.mode == 'zeta':
                self.ui_obj.project_instance.graph.permute_zeta_j_k(i, j, k)
            else:
                logger.warning('Unknown mode. Using district!')
                self.ui_obj.project_instance.graph.permute_j_k(i, j, k)
        self.redraw_neighbourhood(i)
        if center_view:
            pass
            # self.ui_obj.btn_snap_trigger(column_index=i)
            # self.ui_obj.column_selected(i)
            # print('Hello')

    def eval_style(self, i, m):
        vertex_a = self.ui_obj.project_instance.graph.vertices[i]
        vertex_b = self.ui_obj.project_instance.graph.vertices[self.arcs[i][m].j]
        consistent = vertex_b.partner_query(vertex_a.i)
        if vertex_a.zeta == vertex_b.zeta:
            dislocation = True
        else:
            dislocation = False
        return consistent, dislocation

    def re_draw(self):
        """Redraw contents."""
        if self.ui_obj.project_instance is not None:
            self.re_draw_edges()
            self.re_draw_vertices()
            self.re_draw_mesh_details()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        self.interactive_vertex_objects = []
        for vertex in self.ui_obj.project_instance.graph.vertices:
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(
                ui_obj=self.ui_obj,
                vertex=vertex,
                scale_factor=self.scale_factor,
                r=vertex.r
            ))
            if not vertex.void:
                self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self):
        """Redraws all edge elements."""
        for inner_edges in self.arcs:
            for edge_item in inner_edges:
                self.removeItem(edge_item)
        self.arcs = []

        for vertex_a in self.ui_obj.project_instance.graph.vertices:
            self.arcs.append([])
            if self.mode == 'district':
                partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.partners
                )
                out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.out_semi_partners
                )
            elif self.mode == 'zeta':
                partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.local_zeta_map['partners']
                )
                out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.local_zeta_map['out_semi_partners']
                )
            else:
                logger.warning('Unknown mode. Using district!')
                partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.partners
                )
                out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                    vertex_a.out_semi_partners
                )

            for vertex_b in partners:
                p1 = vertex_a.im_pos()
                p2 = vertex_b.im_pos()
                if vertex_a.zeta == vertex_b.zeta:
                    co_planar = True
                else:
                    co_planar = False
                self.arcs[-1].append(GUI_custom_components.Arrow(
                    i=vertex_a.i, j=vertex_b.i, p1=p1, p2=p2,
                    r=self.ui_obj.project_instance.r,
                    scale_factor=self.scale_factor, dual_arc=True,
                    co_planar=co_planar
                ))
                self.addItem(self.arcs[-1][-1])

            for vertex_b in out_semi_partners:
                p1 = vertex_a.im_pos()
                p2 = vertex_b.im_pos()
                self.arcs[-1].append(GUI_custom_components.Arrow(
                    i=vertex_a.i, j=vertex_b.i, p1=p1, p2=p2,
                    r=self.ui_obj.project_instance.r,
                    scale_factor=self.scale_factor, dual_arc=False,
                    co_planar=False
                ))
                self.addItem(self.arcs[-1][-1])
                if not self.ui_obj.control_window.chb_graph.isChecked():
                    self.arcs[-1][-1].hide()

    def re_draw_mesh_details(self):
        """Redraws all mesh details"""
        for mesh_detail in self.mesh_details:
            self.removeItem(mesh_detail)
        self.mesh_details = []

        for mesh in self.ui_obj.project_instance.graph.meshes:
            detail = GUI_custom_components.MeshDetail(mesh=mesh)
            self.addItem(detail)
            self.mesh_details.append(detail)
            if self.ui_obj.control_window.chb_toggle_mesh.isChecked():
                detail.show()
            else:
                detail.hide()

    def redraw_star(self, i):
        for edge_item in self.arcs[i]:
            edge_item.hide()
            self.removeItem(edge_item)
        self.arcs[i] = []
        vertex_a = self.ui_obj.project_instance.graph.vertices[i]
        if self.mode == 'district':
            partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.partners
            )
            out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.out_semi_partners
            )
        elif self.mode == 'zeta':
            partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.local_zeta_map['partners']
            )
            out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.local_zeta_map['out_semi_partners']
            )
        else:
            logger.warning('Unknown mode. Using district!')
            partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.partners
            )
            out_semi_partners = self.ui_obj.project_instance.graph.get_vertex_objects_from_indices(
                vertex_a.out_semi_partners
            )
        for vertex_b in partners:
            p1 = vertex_a.im_pos()
            p1 = (p1[0], p1[1])
            p2 = vertex_b.im_pos()
            p2 = (p2[0], p2[1])
            if vertex_a.zeta == vertex_b.zeta:
                co_planar = True
            else:
                co_planar = False
            self.arcs[i].append(GUI_custom_components.Arrow(
                i=vertex_a.i, j=vertex_b.i, p1=p1, p2=p2,
                r=self.ui_obj.project_instance.r,
                scale_factor=self.scale_factor, dual_arc=True,
                co_planar=co_planar
            ))
            self.addItem(self.arcs[i][-1])
        for vertex_b in out_semi_partners:
            p1 = vertex_a.im_pos()
            p1 = (p1[0], p1[1])
            p2 = vertex_b.im_pos()
            p2 = (p2[0], p2[1])
            self.arcs[i].append(GUI_custom_components.Arrow(
                i=vertex_a.i, j=vertex_b.i, p1=p1, p2=p2,
                r=self.ui_obj.project_instance.r,
                scale_factor=self.scale_factor, dual_arc=False,
                co_planar=False
            ))
            self.addItem(self.arcs[i][-1])
            if not self.ui_obj.control_window.chb_graph.isChecked():
                self.arcs[i][-1].hide()

    def redraw_neighbourhood(self, i):
        for neighbours in self.ui_obj.project_instance.graph.vertices[i].district + [i]:
            self.redraw_star(neighbours)


class InfoGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, scale_factor=1):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.scale_factor = scale_factor
        self.interactive_vertex_objects = []
        self.edges = []
        self.mesh_details = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        if GUI_settings.theme == 'dark':
            self.setBackgroundBrush(GUI_settings.background_brush)
        self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        if self.ui_obj.project_instance is not None:
            self.re_draw_edges()
            self.re_draw_vertices()
            self.re_draw_mesh_details()

    def re_draw_vertices(self):
        pass

    def re_draw_edges(self):
        """Redraws all edge elements."""
        for inner_edges in self.edges:
            for edge_item in inner_edges:
                self.removeItem(edge_item)
        self.edges = []

        for vertex_a in self.ui_obj.project_instance.graph.vertices:
            inner_edges = []
            for n, vertex_b in enumerate(self.ui_obj.project_instance.graph.get_neighbours(vertex_a.i)):
                p1 = vertex_a.real_coor()
                p2 = vertex_b.real_coor()
                real_distance = self.ui_obj.project_instance.graph.real_distance(vertex_a.i, vertex_b.i)
                hard_sphere_distance = self.ui_obj.project_instance.graph.get_hard_sphere_distance(vertex_a.i, vertex_b.i)
                max_shift = 70
                if hard_sphere_distance > real_distance:
                    if hard_sphere_distance - real_distance > max_shift:
                        difference = max_shift
                    else:
                        difference = hard_sphere_distance - real_distance
                    multiplier = - difference / max_shift + 1
                    color = (multiplier * 255, multiplier * 255, 255, 255)
                else:
                    if real_distance - hard_sphere_distance > max_shift:
                        difference = max_shift
                    else:
                        difference = real_distance - hard_sphere_distance
                    multiplier = - difference / max_shift + 1
                    color = (255, multiplier * 255, multiplier * 255, 255)

                inner_edges.append(GUI_custom_components.DistanceArrow(color=color, i=vertex_a.i, j=vertex_b.i, p1=p1,
                                                                       p2=p2, r=self.ui_obj.project_instance.r,
                                                                       scale_factor=self.scale_factor))

                self.addItem(inner_edges[-1])
                if n >= vertex_a.n():
                    inner_edges[-1].hide()
            self.edges.append(inner_edges)

    def re_draw_mesh_details(self):
        """Redraws all mesh details"""
        for mesh_detail in self.mesh_details:
            self.removeItem(mesh_detail)
        self.mesh_details = []

        if not len(self.ui_obj.project_instance.graph.find_intersects()) > 0:
            for mesh in self.ui_obj.project_instance.graph.meshes:
                detail = GUI_custom_components.MeshDetail(mesh=mesh)
                self.addItem(detail)
                self.mesh_details.append(detail)
                if self.ui_obj.control_window.chb_toggle_mesh.isChecked():
                    detail.show()
                else:
                    detail.hide()

    def redraw_star(self, i):
        for edge_item in self.edges[i]:
            self.removeItem(edge_item)
        self.edges[i] = []
        vertex_a = self.ui_obj.project_instance.graph.vertices[i]
        for n, vertex_b in enumerate(self.ui_obj.project_instance.graph.get_neighbours(vertex_a.i)):
            p1 = vertex_a.real_coor()
            p2 = vertex_b.real_coor()
            real_distance = self.ui_obj.project_instance.graph.real_distance(vertex_a.i, vertex_b.i,
                                                                             self.ui_obj.project_instance.scale)
            hard_sphere_distance = self.ui_obj.project_instance.graph.get_hard_sphere_distance(vertex_a.i, vertex_b.i)
            max_shift = 70
            if hard_sphere_distance > real_distance:
                if hard_sphere_distance - real_distance > max_shift:
                    difference = max_shift
                else:
                    difference = hard_sphere_distance - real_distance
                multiplier = - difference / max_shift + 1
                color = (255, multiplier * 255, multiplier * 255, 255)
            else:
                if real_distance - hard_sphere_distance > max_shift:
                    difference = max_shift
                else:
                    difference = real_distance - hard_sphere_distance
                multiplier = - difference / max_shift + 1
                color = (multiplier * 255, multiplier * 255, 255, 255)

            self.edges[i].append(GUI_custom_components.DistanceArrow(color=color, i=vertex_a.i, j=vertex_b.i, p1=p1,
                                                                     p2=p2, r=self.ui_obj.project_instance.r,
                                                                     scale_factor=self.scale_factor))
            self.addItem(self.edges[i][-1])
            if n >= vertex_a.n():
                self.edges[i][-1].hide()

    def redraw_neighbourhood(self, i):
        self.redraw_star(i)
        for neighbours in self.ui_obj.project_instance.graph.vertices[i].neighbour_indices:
            self.redraw_star(neighbours)


class AtomicSubGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, sub_graph=None, scale_factor=1):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic sub-graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.scale_factor = scale_factor
        self.interactive_vertex_objects = []
        self.edges = []
        self.vectors = []
        self.labels = []
        self.background_image = background
        self.report = ''
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        if GUI_settings.theme == 'dark':
            self.setBackgroundBrush(GUI_settings.background_brush)
        self.sub_graph = sub_graph
        if sub_graph is not None:
            self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        if self.ui_obj.project_instance is not None:
            self.re_draw_vertices()
            self.re_draw_edges()
            self.label_angles(angle_vectors=False)
            self.relay_summary()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        for vertex in self.sub_graph.vertices:
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(self.ui_obj, vertex.i, vertex.r, self.scale_factor))
            self.interactive_vertex_objects[-1].setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            self.interactive_vertex_objects[-1].setFlag(QtWidgets.QGraphicsItem.ItemIsPanel, True)
            self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self):
        """Redraws all edge elements."""
        for edge in self.sub_graph.arcs:
            consistent = edge.dual_arc
            dislocation = edge.co_planar
            p1 = edge.vertex_a.im_pos()
            p2 = edge.vertex_b.im_pos()
            self.edges.append(GUI_custom_components.Arrow(i=edge.vertex_a.i, j=edge.vertex_b.i, p1=p1, p2=p2, r=self.ui_obj.project_instance.r, scale_factor=self.scale_factor, dual_arc=consistent, co_planar=dislocation))
            self.addItem(self.edges[-1])

    def label_angles(self, angle_vectors=False):
        """Label all sub-graph angles."""
        self.report += 'Sub-graph centered on vertex {}:----------\n'.format(self.sub_graph.vertex_indices[0])
        for m, mesh in enumerate(self.sub_graph.meshes):
            self.report += 'Mesh {}:\n'.format(m)
            self.report += '    Number of corners: {}\n'.format(str(mesh.order))
            self.report += '    Sum of angles: {}\n'.format(str(sum(mesh.angles)))
            self.report += '    Variance of angles: {}\n'.format(utils.variance(mesh.angles))
            self.report += '    Symmetry prob vector from central angle: {}\n'.format(str([0, 0, 0]))
            self.report += '    corners: {}\n'.format(mesh.vertex_indices)
            self.report += '    Angles:\n'
            for i, corner in enumerate(mesh.vertices):
                self.report += '        a{}{} = {}\n'.format(m, i, mesh.angles[i])
                p1 = corner.im_pos()
                p1 = (p1[0], p1[1])
                p2 = (p1[0] + 0.5 * corner.r * mesh.angle_vectors[i][0], p1[1] + 0.5 * corner.r * mesh.angle_vectors[i][1])

                if angle_vectors:
                    self.vectors.append(GUI_custom_components.Arrow(p1, p2, corner.r, self.scale_factor, False, False))
                    self.addItem(self.vectors[-1].arrow[0])

                angle_text = QtWidgets.QGraphicsSimpleTextItem()
                angle_text.setText('a{}{}'.format(m, i))
                angle_text.setFont(GUI_settings.font_tiny)
                rect = angle_text.boundingRect()
                angle_text.setPos(self.scale_factor * p2[0] - 0.5 * rect.width(), self.scale_factor * p2[1] - 0.5 * rect.height())
                self.addItem(angle_text)

    def relay_summary(self):
        logger.info(self.report)


class AntiGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, scale_factor=1, graph=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.scale_factor = scale_factor
        self.graph = graph
        self.interactive_vertex_objects = []
        self.edges = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        if GUI_settings.theme == 'dark':
            self.setBackgroundBrush(GUI_settings.background_brush)
        self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        if self.graph is not None:
            self.re_draw_edges()
            self.re_draw_vertices()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        self.interactive_vertex_objects = []
        for vertex in self.graph.get_non_void_vertices():
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(self.ui_obj, vertex.i, vertex.r, self.scale_factor))
            self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self):
        """Redraws all edge elements."""
        self.graph.map_arcs()
        for edge_item in self.edges:
            self.removeItem(edge_item)
        self.edges = []
        for edge in self.graph.arcs:
            p1 = edge.vertex_a.im_pos()
            p1 = (p1[0], p1[1])
            p2 = edge.vertex_b.im_pos()
            p2 = (p2[0], p2[1])
            real_distance = edge.spatial_separation
            hard_sphere_distance = edge.hard_sphere_separation
            max_shift = 70
            if hard_sphere_distance > real_distance:
                if hard_sphere_distance - real_distance > max_shift:
                    difference = max_shift
                else:
                    difference = hard_sphere_distance - real_distance
                multiplier = - difference / max_shift + 1
                color = (multiplier * 255, multiplier * 255, 255, 255)
            else:
                if real_distance - hard_sphere_distance > max_shift:
                    difference = max_shift
                else:
                    difference = real_distance - hard_sphere_distance
                multiplier = - difference / max_shift + 1
                color = (255, multiplier * 255, multiplier * 255, 255)

            self.edges.append(GUI_custom_components.DistanceArrow(color=color, i=edge.vertex_a.i, j=edge.vertex_b.i,
                                                                  p1=p1, p2=p2, r=self.ui_obj.project_instance.r,
                                                                  scale_factor=self.scale_factor))
            self.addItem(self.edges[-1])
            if edge.vertex_a.zeta == 1 and edge.vertex_b.zeta == 1:
                pass
            else:
                self.edges[-1].setZValue(-2)

    def toggle_level_0(self, on):
        for vertex in self.graph.get_non_void_vertices():
            if vertex.zeta == 0:
                if on:
                    self.interactive_vertex_objects[vertex.i].show()
                else:
                    self.interactive_vertex_objects[vertex.i].hide()
        for m, edge in enumerate(self.graph.arcs):
            if edge.vertex_a.zeta == 0 and edge.vertex_b.zeta == 0:
                if on:
                    self.edges[m].show()
                else:
                    self.edges[m].hide()

    def toggle_level_1(self, on):
        for vertex in self.graph.get_non_void_vertices():
            if vertex.zeta == 1:
                if on:
                    self.interactive_vertex_objects[vertex.i].show()
                else:
                    self.interactive_vertex_objects[vertex.i].hide()
        for m, edge in enumerate(self.graph.arcs):
            if edge.vertex_a.zeta == 1 and edge.vertex_b.zeta == 1:
                if on:
                    self.edges[m].show()
                else:
                    self.edges[m].hide()


class ZoomGraphicsView(QtWidgets.QGraphicsView):
    """An adaptation of QtWidgets.QGraphicsView that supports zooming"""

    def __init__(self, parent=None, ui_obj=None, trigger_func=None, tab_index=0):
        super(ZoomGraphicsView, self).__init__(parent)
        self.ui_obj = ui_obj
        self.trigger_func = trigger_func
        self.tab_index = tab_index

    def translate_w(self, amount, relay=True):

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self.translate(0, amount)

        if self.ui_obj.lock_views and relay:
            for i, zgv in enumerate(self.ui_obj.gv_list):
                if not i == self.tab_index:
                    zgv.translate_w(amount, relay=False)

    def translate_s(self, amount, relay=True):

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self.translate(0, -amount)

        if self.ui_obj.lock_views and relay:
            for i, zgv in enumerate(self.ui_obj.gv_list):
                if not i == self.tab_index:
                    zgv.translate_s(amount, relay=False)

    def translate_d(self, amount, relay=True):

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self.translate(-amount, 0)

        if self.ui_obj.lock_views and relay:
            for i, zgv in enumerate(self.ui_obj.gv_list):
                if not i == self.tab_index:
                    zgv.translate_d(amount, relay=False)

    def translate_a(self, amount, relay=True):

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self.translate(amount, 0)

        if self.ui_obj.lock_views and relay:
            for i, zgv in enumerate(self.ui_obj.gv_list):
                if not i == self.tab_index:
                    zgv.translate_a(amount, relay=False)

    def wheelEvent(self, event, relay=True):

        modifier = QtWidgets.QApplication.keyboardModifiers()
        if modifier == QtCore.Qt.ShiftModifier:

            # Zoom Factor
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor

            # Set Anchors
            self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

            # Save the scene pos
            oldPos = self.mapToScene(event.pos())

            # Zoom
            if event.angleDelta().y() > 0:
                zoomFactor = zoom_in_factor
            else:
                zoomFactor = zoom_out_factor
            self.scale(zoomFactor, zoomFactor)

            # Get the new position
            newPos = self.mapToScene(event.pos())

            # Move scene to old position
            delta = newPos - oldPos
            self.translate(delta.x(), delta.y())

        else:

            super(ZoomGraphicsView, self).wheelEvent(event)

        if self.ui_obj.lock_views and relay:
            for i, zgv in enumerate(self.ui_obj.gv_list):
                if not i == self.tab_index:
                    zgv.wheelEvent(event, relay=False)

    def keyPressEvent(self, event):
        if self.trigger_func is not None:
            self.trigger_func(event.key())


# ----------
# Custom Main window tools:
# ----------


class TerminalTextEdit(QtWidgets.QPlainTextEdit):

    def __init__(self, *args):
        super().__init__(*args)

        self.setReadOnly(True)
        self.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)


class TerminalHandler(logging.Handler):

    def __init__(self):
        super().__init__()

        self.widget = TerminalTextEdit()

        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(funcName)s:%(message)s'))

    def set_mode(self, debug=False):

        if debug:
            self.setLevel(logging.DEBUG)
            self.setFormatter(logging.Formatter('%(name)s: %(levelname)s: %(funcName)s: %(message)s'))
        else:
            self.setLevel(logging.INFO)
            self.setFormatter(logging.Formatter('%(name)s: %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
        QtWidgets.QApplication.processEvents()


class Terminal(QtWidgets.QWidget):

    def __init__(self, *args, obj=None):
        super().__init__(*args)

        self.ui_obj = obj

        self.handler = TerminalHandler()

        self.btn_save_log = GUI_custom_components.SmallButton('Save log', self, trigger_func=self.ui_obj.btn_save_log_trigger)
        self.btn_clear_log = GUI_custom_components.SmallButton('Clear log', self, trigger_func=self.ui_obj.btn_clear_log_trigger)
        self.btn_eval = GUI_custom_components.SmallButton('Eval', self, trigger_func=self.eval_input)
        self.btn_eval.setDisabled(True)

        self.chb_input = QtWidgets.QCheckBox('Enable input (advanced)')
        self.chb_input.setChecked(False)
        self.chb_input.toggled.connect(self.set_input)

        self.terminal_btns_layout = QtWidgets.QHBoxLayout()
        self.terminal_btns_layout.addWidget(self.btn_save_log)
        self.terminal_btns_layout.addWidget(self.btn_clear_log)
        self.terminal_btns_layout.addWidget(self.chb_input)
        self.terminal_btns_layout.addStretch()

        self.eval_text = QtWidgets.QLineEdit()
        self.eval_text.setDisabled(True)
        self.input_layout = QtWidgets.QHBoxLayout()
        self.input_layout.addWidget(self.eval_text)
        self.input_layout.addWidget(self.btn_eval)

        self.terminal_display_layout = QtWidgets.QVBoxLayout()
        self.terminal_display_layout.addLayout(self.terminal_btns_layout)
        self.terminal_display_layout.addWidget(self.handler.widget)
        self.terminal_display_layout.addLayout(self.input_layout)

        self.setLayout(self.terminal_display_layout)

        # Set tooltips:
        self.mode_tooltip(self.ui_obj.menu.toggle_tooltips_action.isChecked())

    def mode_tooltip(self, on):
        if on:
            self.btn_save_log.setToolTip(GUI_tooltips.btn_save_log)
            self.btn_clear_log.setToolTip(GUI_tooltips.btn_clear_log)
        else:
            self.btn_save_log.setToolTip('')
            self.btn_clear_log.setToolTip('')

    def set_input(self, state):
        self.eval_text.setDisabled(not state)
        self.btn_eval.setDisabled(not state)

    def eval_input(self):
        if self.chb_input.isChecked():
            self.ui_obj.btn_eval_trigger(self.eval_text.text())


class ControlWindow(QtWidgets.QWidget):

    def __init__(self, *args, obj=None):
        super().__init__(*args)

        self.ui_obj = obj

        # Prob. Vector graphic:
        self.height = 130
        self.width = 160

        self.probGraphicView = QtWidgets.QGraphicsView()

        self.probGraphicView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.probGraphicView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.probGraphicView.setMinimumHeight(self.height)
        self.probGraphicView.setMaximumHeight(self.height)
        self.probGraphicView.setMinimumWidth(self.width)
        self.probGraphicView.setMaximumWidth(self.width)
        if GUI_settings.theme == 'dark':
            self.probGraphicView.setBackgroundBrush(GUI_settings.background_brush)

        self.probGraphicLayout = QtWidgets.QHBoxLayout()
        self.probGraphicLayout.addWidget(self.probGraphicView)
        self.probGraphicLayout.addStretch()

        self.draw_histogram()

        # -------------------------------------------------------------------------
        # Define control widgets:
        # -------------------------------------------------------------------------

        # - Project controls
        # -- Labels
        self.lbl_filename = QtWidgets.QLabel('Filename: ')
        self.lbl_location = QtWidgets.QLabel('Location: ')
        self.lbl_model = QtWidgets.QLabel('Associated model: ')
        # -- Set buttons
        self.btn_set_model_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_model_trigger, label=self.lbl_model)
        # -- Buttons
        self.btn_new_project = GUI_custom_components.MediumButton('Import dm3', self, trigger_func=self.ui_obj.btn_new_project_trigger)
        self.btn_open_project = GUI_custom_components.MediumButton('Open', self, trigger_func=self.ui_obj.btn_open_project_trigger)
        self.btn_save_project = GUI_custom_components.MediumButton('Save', self, trigger_func=self.ui_obj.btn_save_project_trigger)
        self.btn_close_project = GUI_custom_components.MediumButton('Close', self, trigger_func=self.ui_obj.btn_close_project_trigger)
        self.btn_configure_classes = GUI_custom_components.MediumButton('Categories', self, trigger_func=self.ui_obj.btn_configure_classes_trigger)

        # - Image controls
        # -- Labels
        self.lbl_image_width = QtWidgets.QLabel('Image width (pixels): ')
        self.lbl_image_height = QtWidgets.QLabel('Image height (pixels): ')
        self.lbl_upsampled = QtWidgets.QLabel('Automatic bilinear upsampling performed: ')
        # -- Checkboxes
        self.chb_lock_views = QtWidgets.QCheckBox('Lock views')
        # -- Set buttons
        # -- Buttons
        self.btn_show_stats = GUI_custom_components.MediumButton('Stats', self, trigger_func=self.ui_obj.btn_show_stats_trigger)
        self.btn_show_source = GUI_custom_components.MediumButton('Source', self, trigger_func=self.ui_obj.btn_view_image_title_trigger)
        self.btn_align_views = GUI_custom_components.MediumButton('Align views', self, trigger_func=self.ui_obj.btn_align_views_trigger)

        # - Column detection controls
        # -- Labels
        self.lbl_atomic_radii = QtWidgets.QLabel('Approx atomic radii (pixels): ')
        self.lbl_overhead_radii = QtWidgets.QLabel('Overhead (pixels): ')
        self.lbl_pixel_average = QtWidgets.QLabel('Average pixel value: ')
        self.lbl_detection_threshold = QtWidgets.QLabel('Detection threshold value: ')
        self.lbl_search_matrix_peak = QtWidgets.QLabel('Search matrix peak: ')
        self.lbl_search_size = QtWidgets.QLabel('Search size: ')
        self.lbl_scale = QtWidgets.QLabel('Scale (pm / pixel): ')
        self.lbl_num_detected_columns = QtWidgets.QLabel('Number of detected columns: ')
        # -- Checkboxes
        self.chb_plot_results = QtWidgets.QCheckBox('Plot column detection summary')
        self.chb_automatic_threshold = QtWidgets.QCheckBox('Use automatic thresholding')
        self.chb_toggle_positions = QtWidgets.QCheckBox('Show column position overlay')
        # -- Set buttons
        self.btn_set_threshold_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_threshold_trigger, label=self.lbl_detection_threshold)
        self.btn_set_search_size_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_search_size_trigger, label=self.lbl_search_size)
        self.btn_set_scale_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_scale_trigger, label=self.lbl_scale)
        # -- Buttons
        self.btn_start_alg_1 = GUI_custom_components.MediumButton('Start', self, trigger_func=self.ui_obj.btn_continue_detection_trigger)
        self.btn_reset_alg_1 = GUI_custom_components.MediumButton('Reset', self, trigger_func=self.ui_obj.btn_restart_detection_trigger)
        self.btn_redraw_search_mat = GUI_custom_components.MediumButton('Redraw search mat', self, trigger_func=self.ui_obj.btn_redraw_search_mat_trigger)

        # - Column characterization controls
        # -- Labels
        self.lbl_alloy = QtWidgets.QLabel('Alloy: ')
        # -- Checkboxes
        self.chb_show_graphic_updates = QtWidgets.QCheckBox('Show graphic updates (slow)')
        # -- Set buttons
        self.btn_set_alloy_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_alloy_trigger, label=self.lbl_alloy)
        # -- Buttons
        self.btn_start_alg_2 = GUI_custom_components.MediumButton('Start', self, trigger_func=self.ui_obj.btn_continue_analysis_trigger)
        self.btn_reset_alg_2 = GUI_custom_components.MediumButton('Reset', self, trigger_func=self.ui_obj.btn_restart_analysis_trigger)
        self.btn_invert_lvl_alg_2 = GUI_custom_components.MediumButton('Invert lvl', self, trigger_func=self.ui_obj.btn_invert_levels_trigger)

        # - Selected column controls
        # -- Labels
        self.lbl_column_index = QtWidgets.QLabel('Column index: ')
        self.lbl_column_x_pos = QtWidgets.QLabel('x: ')
        self.lbl_column_y_pos = QtWidgets.QLabel('y: ')
        self.lbl_column_peak_gamma = QtWidgets.QLabel('Peak gamma: ')
        self.lbl_column_avg_gamma = QtWidgets.QLabel('Avg gamma: ')
        self.lbl_column_species = QtWidgets.QLabel('Atomic species: ')
        self.lbl_advanced_species = QtWidgets.QLabel('Advanced species: ')
        self.lbl_column_level = QtWidgets.QLabel('Zeta: ')
        self.lbl_prob_vector = QtWidgets.QLabel('Probability histogram: ')
        self.lbl_central_variance = QtWidgets.QLabel('Theta variance: ')
        self.lbl_alpha_max = QtWidgets.QLabel('Alpha max: ')
        self.lbl_alpha_min = QtWidgets.QLabel('Alpha min: ')
        self.lbl_theta_max = QtWidgets.QLabel('Theta max: ')
        self.lbl_theta_min = QtWidgets.QLabel('Theta min: ')
        self.lbl_theta_mean = QtWidgets.QLabel('Theta mean: ')
        self.lbl_norm_peak_gamma = QtWidgets.QLabel('Normalized peak gamma: ')
        self.lbl_norm_avg_gamma = QtWidgets.QLabel('Normalized avg gamma: ')
        # -- Checkboxes
        self.chb_precipitate_column = QtWidgets.QCheckBox('Precipitate column')
        self.chb_show = QtWidgets.QCheckBox('Show in overlay')
        self.chb_move = QtWidgets.QCheckBox('Enable move')
        # -- Set buttons
        self.btn_find_column_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_find_column_trigger, label=self.lbl_column_index)
        self.btn_set_advanced_species_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_advanced_species_trigger, label=self.lbl_advanced_species)
        self.btn_set_level_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_level_trigger, label=self.lbl_column_level)
        # -- Buttons
        self.btn_cancel_move = GUI_custom_components.SmallButton('Cancel', self, trigger_func=self.ui_obj.btn_cancel_move_trigger)
        self.btn_set_move = GUI_custom_components.SmallButton('Accept', self, trigger_func=self.ui_obj.btn_set_position_trigger)
        self.btn_set_variant = GUI_custom_components.MediumButton('Set variant', self, trigger_func=self.ui_obj.btn_set_variant_trigger)
        self.btn_delete = GUI_custom_components.MediumButton('Delete', self, trigger_func=self.ui_obj.btn_delete_trigger)
        self.btn_print_details = GUI_custom_components.MediumButton('Print', self, trigger_func=self.ui_obj.btn_print_details_trigger)
        self.btn_snap = GUI_custom_components.MediumButton('Show', self, trigger_func=self.ui_obj.btn_snap_trigger)
        self.btn_deselect = GUI_custom_components.MediumButton('Deselect', self, trigger_func=self.ui_obj.btn_deselect_trigger)
        self.btn_new = GUI_custom_components.MediumButton('New', self, trigger_func=self.ui_obj.btn_new_column_trigger)

        # - Atomic graph controls
        # -- Labels
        self.lbl_chi = QtWidgets.QLabel('Chi: ')
        self.lbl_average_degree = QtWidgets.QLabel('Average degree: ')
        self.lbl_sub_graph_type = QtWidgets.QLabel('Sub-graph type: ')
        self.lbl_sub_graph_order = QtWidgets.QLabel('Sub-graph order: ')
        # -- Checkboxes
        self.chb_perturb_mode = QtWidgets.QCheckBox('Enable permute mode')
        self.chb_enable_ruler = QtWidgets.QCheckBox('Enable ruler')
        self.chb_graph = QtWidgets.QCheckBox('Show inconsistent connections')
        self.chb_toggle_mesh = QtWidgets.QCheckBox('Show mesh details')
        self.chb_show_level_0 = QtWidgets.QCheckBox('Show level 0 plane')
        self.chb_show_level_1 = QtWidgets.QCheckBox('Show level 1 plane')
        # -- Set buttons
        self.btn_set_sub_graph_type_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_sub_graph_type_trigger, label=self.lbl_sub_graph_type)
        self.btn_set_sub_graph_order_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_sub_graph_order_trigger, label=self.lbl_sub_graph_order)
        # -- Buttons
        self.btn_sub = GUI_custom_components.MediumButton('Build sub-graph', self, trigger_func=self.ui_obj.btn_gen_sub_graph)
        self.btn_refresh_graph = GUI_custom_components.MediumButton('Refresh', self, trigger_func=self.ui_obj.btn_refresh_graph_trigger)
        self.btn_refresh_mesh = GUI_custom_components.MediumButton('Refresh mesh', self, trigger_func=self.ui_obj.btn_refresh_mesh_trigger)
        self.btn_build_anti_graph = GUI_custom_components.MediumButton('Build anti-graph', self, trigger_func=self.ui_obj.btn_build_anti_graph_trigger)
        self.btn_build_info_graph = GUI_custom_components.MediumButton('Build info-graph', self, trigger_func=self.ui_obj.btn_build_info_graph_trigger)

        # - Data and export controls
        # -- Buttons
        self.btn_export = GUI_custom_components.MediumButton('Stats', self, trigger_func=self.ui_obj.menu_export_data_trigger)
        self.btn_plot = GUI_custom_components.MediumButton('Make plots', self, trigger_func=self.ui_obj.btn_make_plot_trigger)
        self.btn_print_distances = GUI_custom_components.MediumButton('Print distances', self, trigger_func=self.ui_obj.btn_print_distances_trigger)
        self.btn_pca = GUI_custom_components.MediumButton('Perform PCA', self, trigger_func=self.ui_obj.btn_pca_trigger)
        self.btn_calc_models = GUI_custom_components.MediumButton('Calculate model', self, trigger_func=self.ui_obj.btn_calc_models_trigger)
        self.btn_plot_models = GUI_custom_components.MediumButton('Plot model', self, trigger_func=self.ui_obj.btn_plot_models_trigger)

        # - Overlay controls
        # -- Checkboxes
        self.chb_raw_image = QtWidgets.QCheckBox('Raw image')
        self.chb_black_background = QtWidgets.QCheckBox('Black background')
        self.chb_structures = QtWidgets.QCheckBox('Structures')
        self.chb_edge = QtWidgets.QCheckBox('Edge columns')
        self.chb_si_columns = QtWidgets.QCheckBox('Si columns')
        self.chb_si_network = QtWidgets.QCheckBox('Si network')
        self.chb_mg_columns = QtWidgets.QCheckBox('Mg columns')
        self.chb_particle = QtWidgets.QCheckBox('Particle')
        self.chb_al_columns = QtWidgets.QCheckBox('Al columns')
        self.chb_al_network = QtWidgets.QCheckBox('Al network')
        self.chb_cu_columns = QtWidgets.QCheckBox('Cu columns')
        self.chb_cu_network = QtWidgets.QCheckBox('Cu network')
        self.chb_ag_columns = QtWidgets.QCheckBox('Ag columns')
        self.chb_ag_network = QtWidgets.QCheckBox('Ag network')
        self.chb_un_columns = QtWidgets.QCheckBox('Un columns')
        self.chb_columns = QtWidgets.QCheckBox('Columns')
        self.chb_al_mesh = QtWidgets.QCheckBox('Al-mesh')
        self.chb_neighbours = QtWidgets.QCheckBox('Neighbour path')
        self.chb_legend = QtWidgets.QCheckBox('Legend')
        self.chb_scalebar = QtWidgets.QCheckBox('Scalebar')
        self.chb_0_plane = QtWidgets.QCheckBox('0 Plane')
        self.chb_1_plane = QtWidgets.QCheckBox('1/2 Plane')
        # -- Buttons
        self.btn_set_style = GUI_custom_components.MediumButton('Configure', self, trigger_func=self.ui_obj.btn_set_style_trigger)
        self.btn_show_all = GUI_custom_components.MediumButton('Show all', self, trigger_func=self.ui_obj.btn_show_all_trigger)
        self.btn_hide_all = GUI_custom_components.MediumButton('Hide all', self, trigger_func=self.ui_obj.btn_hide_all_trigger)

        # - Debug controls
        # -- Labels
        self.lbl_starting_index = QtWidgets.QLabel('Default starting index: ')
        # -- Set buttons
        self.btn_set_start_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_start_trigger,label=self.lbl_starting_index)
        # -- Buttons
        self.btn_test = GUI_custom_components.MediumButton('Test', self, trigger_func=self.ui_obj.btn_test_trigger)
        self.btn_crash = GUI_custom_components.MediumButton('Crash program', self, trigger_func=self.ui_obj.btn_crash_trigger)
        self.btn_test_data_manager = GUI_custom_components.MediumButton('Test data-manager', self, trigger_func=self.ui_obj.btn_test_data_manager_trigger)
        self.btn_show_model_prediction = GUI_custom_components.MediumButton('Show model prediction', self, trigger_func=self.ui_obj.btn_show_model_prediction_trigger)

        # -------------------------------------------------------------------------
        # Internal layouts
        # -------------------------------------------------------------------------

        # - Project controls
        btn_project_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btn_new_project)
        layout.addWidget(self.btn_open_project)
        layout.addWidget(self.btn_save_project)
        layout.addStretch()
        btn_project_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btn_close_project)
        layout.addWidget(self.btn_configure_classes)
        layout.addStretch()
        btn_project_layout.addLayout(layout)

        # - Image controls
        btn_image_btns_layout = QtWidgets.QHBoxLayout()
        btn_image_btns_layout.addWidget(self.btn_show_stats)
        btn_image_btns_layout.addWidget(self.btn_show_source)
        btn_image_btns_layout.addWidget(self.btn_align_views)
        btn_image_btns_layout.addStretch()

        # - Column detection controls
        btn_alg_1_btns_layout = QtWidgets.QHBoxLayout()
        btn_alg_1_btns_layout.addWidget(self.btn_start_alg_1)
        btn_alg_1_btns_layout.addWidget(self.btn_reset_alg_1)
        btn_alg_1_btns_layout.addWidget(self.btn_redraw_search_mat)
        btn_alg_1_btns_layout.addStretch()

        # - Column characterization controls
        btn_alg_2_btns_layout = QtWidgets.QHBoxLayout()
        btn_alg_2_btns_layout.addWidget(self.btn_start_alg_2)
        btn_alg_2_btns_layout.addWidget(self.btn_reset_alg_2)
        btn_alg_2_btns_layout.addWidget(self.btn_invert_lvl_alg_2)
        btn_alg_2_btns_layout.addStretch()

        # - Selected column controls
        btn_move_control_layout = QtWidgets.QHBoxLayout()
        btn_move_control_layout.addWidget(self.chb_move)
        btn_move_control_layout.addWidget(self.btn_cancel_move)
        btn_move_control_layout.addWidget(self.btn_set_move)
        btn_move_control_layout.addStretch()
        btn_column_btns_layout_1 = QtWidgets.QHBoxLayout()
        btn_column_btns_layout_1.addWidget(self.btn_new)
        btn_column_btns_layout_1.addWidget(self.btn_deselect)
        btn_column_btns_layout_1.addWidget(self.btn_delete)
        btn_column_btns_layout_1.addStretch()
        btn_column_btns_layout_2 = QtWidgets.QHBoxLayout()
        btn_column_btns_layout_2.addWidget(self.btn_print_details)
        btn_column_btns_layout_2.addWidget(self.btn_snap)
        btn_column_btns_layout_2.addWidget(self.btn_set_variant)
        btn_column_btns_layout_2.addStretch()
        btn_column_btns_layout = QtWidgets.QVBoxLayout()
        btn_column_btns_layout.addLayout(btn_column_btns_layout_1)
        btn_column_btns_layout.addLayout(btn_column_btns_layout_2)

        # - Atomic graph controls
        btn_graph_btns_layout = QtWidgets.QHBoxLayout()
        btn_graph_btns_layout.addWidget(self.btn_refresh_graph)
        btn_graph_btns_layout.addWidget(self.btn_print_distances)
        btn_graph_btns_layout.addWidget(self.btn_refresh_mesh)
        btn_graph_btns_layout.addStretch()
        btn_sub_graphs_layout = QtWidgets.QHBoxLayout()
        btn_sub_graphs_layout.addWidget(self.btn_sub)
        btn_sub_graphs_layout.addStretch()
        btn_anti_graph_layout = QtWidgets.QHBoxLayout()
        btn_anti_graph_layout.addWidget(self.btn_build_anti_graph)
        btn_anti_graph_layout.addStretch()
        btn_info_graph_layout = QtWidgets.QHBoxLayout()
        btn_info_graph_layout.addWidget(self.btn_build_info_graph)
        btn_info_graph_layout.addStretch()

        # - Data and export controls
        btn_analysis_layout_1 = QtWidgets.QHBoxLayout()
        btn_analysis_layout_1.addWidget(self.btn_plot)
        btn_analysis_layout_1.addWidget(self.btn_pca)
        btn_analysis_layout_1.addWidget(self.btn_export)
        btn_analysis_layout_1.addStretch()
        btn_analysis_layout_2 = QtWidgets.QHBoxLayout()
        btn_analysis_layout_2.addWidget(self.btn_calc_models)
        btn_analysis_layout_2.addWidget(self.btn_plot_models)
        btn_analysis_layout_2.addStretch()
        btn_analysis_layout = QtWidgets.QVBoxLayout()
        btn_analysis_layout.addLayout(btn_analysis_layout_1)
        btn_analysis_layout.addLayout(btn_analysis_layout_2)

        # - Overlay controls
        btn_overlay_btns_layout = QtWidgets.QHBoxLayout()
        btn_overlay_btns_layout.addWidget(self.btn_set_style)
        btn_overlay_btns_layout.addWidget(self.btn_show_all)
        btn_overlay_btns_layout.addWidget(self.btn_hide_all)
        btn_overlay_btns_layout.addStretch()
        overlay_layout_left = QtWidgets.QVBoxLayout()
        overlay_layout_left.addWidget(self.chb_raw_image)
        overlay_layout_left.addWidget(self.chb_structures)
        overlay_layout_left.addWidget(self.chb_si_columns)
        overlay_layout_left.addWidget(self.chb_cu_columns)
        overlay_layout_left.addWidget(self.chb_al_columns)
        overlay_layout_left.addWidget(self.chb_ag_columns)
        overlay_layout_left.addWidget(self.chb_mg_columns)
        overlay_layout_left.addWidget(self.chb_un_columns)
        overlay_layout_left.addWidget(self.chb_ag_network)
        overlay_layout_left.addWidget(self.chb_legend)
        overlay_layout_left.addWidget(self.chb_scalebar)
        overlay_layout_right = QtWidgets.QVBoxLayout()
        overlay_layout_right.addWidget(self.chb_black_background)
        overlay_layout_right.addWidget(self.chb_edge)
        overlay_layout_right.addWidget(self.chb_si_network)
        overlay_layout_right.addWidget(self.chb_cu_network)
        overlay_layout_right.addWidget(self.chb_al_network)
        overlay_layout_right.addWidget(self.chb_al_mesh)
        overlay_layout_right.addWidget(self.chb_particle)
        overlay_layout_right.addWidget(self.chb_columns)
        overlay_layout_right.addWidget(self.chb_neighbours)
        overlay_layout_right.addWidget(self.chb_0_plane)
        overlay_layout_right.addWidget(self.chb_1_plane)
        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_layout.addLayout(overlay_layout_left)
        overlay_layout.addLayout(overlay_layout_right)
        overlay_layout.addStretch()

        # - Debug controls
        btn_debug_btns_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btn_test_data_manager)
        layout.addWidget(self.btn_test)
        layout.addWidget(self.btn_crash)
        layout.addStretch()
        btn_debug_btns_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btn_show_model_prediction)
        layout.addStretch()
        btn_debug_btns_layout.addLayout(layout)

        # -------------------------------------------------------------------------
        # Set up default behaviour
        # -------------------------------------------------------------------------

        # - Project controls

        # - Image controls
        self.chb_lock_views.setChecked(False)

        # - Column detection controls
        self.chb_plot_results.setChecked(False)
        self.chb_automatic_threshold.setChecked(False)
        self.chb_toggle_positions.setChecked(True)

        # - Column characterization controls
        self.chb_show_graphic_updates.setChecked(False)

        # - Selected column controls
        self.chb_precipitate_column.setChecked(False)
        self.chb_show.setChecked(False)
        self.chb_move.setChecked(False)
        self.btn_cancel_move.setDisabled(True)
        self.btn_set_move.setDisabled(True)

        # - Atomic graph controls
        self.chb_perturb_mode.setChecked(False)
        self.chb_enable_ruler.setChecked(False)
        self.chb_graph.setChecked(True)
        self.chb_toggle_mesh.setChecked(False)
        self.chb_show_level_0.setChecked(True)
        self.chb_show_level_1.setChecked(True)

        # - Overlay controls
        self.chb_raw_image.setChecked(True)
        self.chb_black_background.setChecked(True)
        self.chb_structures.setChecked(True)
        self.chb_edge.setChecked(True)
        self.chb_si_columns.setChecked(True)
        self.chb_si_network.setChecked(False)
        self.chb_cu_columns.setChecked(True)
        self.chb_cu_network.setChecked(False)
        self.chb_al_columns.setChecked(True)
        self.chb_al_network.setChecked(False)
        self.chb_ag_columns.setChecked(True)
        self.chb_ag_network.setChecked(False)
        self.chb_mg_columns.setChecked(True)
        self.chb_particle.setChecked(True)
        self.chb_un_columns.setChecked(True)
        self.chb_columns.setChecked(True)
        self.chb_al_mesh.setChecked(True)
        self.chb_neighbours.setChecked(False)
        self.chb_legend.setChecked(True)
        self.chb_scalebar.setChecked(True)
        self.chb_0_plane.setChecked(True)
        self.chb_1_plane.setChecked(True)

        # - Debug controls

        # -------------------------------------------------------------------------
        # Connect trigger functions:
        # -------------------------------------------------------------------------

        # - Project controls

        # - Image controls
        self.chb_lock_views.toggled.connect(self.ui_obj.chb_lock_views_trigger)

        # - Column detection controls
        self.chb_plot_results.toggled.connect(self.ui_obj.chb_plot_results_trigger)
        self.chb_automatic_threshold.toggled.connect(self.ui_obj.chb_automatic_threshold_trigger)
        self.chb_toggle_positions.toggled.connect(self.ui_obj.chb_toggle_positions_trigger)

        # - Column characterization controls
        self.chb_show_graphic_updates.toggled.connect(self.ui_obj.chb_show_graphic_updates_trigger)

        # - Selected column controls
        self.chb_precipitate_column.toggled.connect(self.ui_obj.chb_precipitate_column_trigger)
        self.chb_show.toggled.connect(self.ui_obj.chb_show_trigger)
        self.chb_move.toggled.connect(self.ui_obj.chb_enable_move_trigger)

        # - Atomic graph controls
        self.chb_perturb_mode.toggled.connect(self.ui_obj.chb_set_perturb_mode_trigger)
        self.chb_graph.toggled.connect(self.ui_obj.chb_graph_detail_trigger)
        self.chb_toggle_mesh.toggled.connect(self.ui_obj.chb_toggle_mesh_trigger)
        self.chb_show_level_0.toggled.connect(self.ui_obj.chb_show_level_0_trigger)
        self.chb_show_level_1.toggled.connect(self.ui_obj.chb_show_level_1_trigger)

        # - Data and export controls

        # - Overlay controls
        self.chb_raw_image.toggled.connect(self.ui_obj.chb_raw_image_trigger)
        self.chb_black_background.toggled.connect(self.ui_obj.chb_black_background_trigger)
        self.chb_structures.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_edge.toggled.connect(self.ui_obj.chb_edge_trigger)
        self.chb_si_columns.toggled.connect(self.ui_obj.chb_toggle_si_trigger)
        self.chb_si_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_cu_columns.toggled.connect(self.ui_obj.chb_toggle_cu_trigger)
        self.chb_cu_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_al_columns.toggled.connect(self.ui_obj.chb_toggle_al_trigger)
        self.chb_al_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_ag_columns.toggled.connect(self.ui_obj.chb_toggle_ag_trigger)
        self.chb_ag_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_mg_columns.toggled.connect(self.ui_obj.chb_toggle_mg_trigger)
        self.chb_particle.toggled.connect(self.ui_obj.chb_particle_trigger)
        self.chb_un_columns.toggled.connect(self.ui_obj.chb_toggle_un_trigger)
        self.chb_columns.toggled.connect(self.ui_obj.chb_toggle_all_trigger)
        self.chb_al_mesh.toggled.connect(self.ui_obj.chb_matrix_trigger)
        self.chb_neighbours.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_legend.toggled.connect(self.ui_obj.chb_legend_trigger)
        self.chb_scalebar.toggled.connect(self.ui_obj.chb_scale_bar_trigger)
        self.chb_0_plane.toggled.connect(self.ui_obj.chb_0_plane_trigger)
        self.chb_1_plane.toggled.connect(self.ui_obj.chb_1_plane_trigger)

        # - Debug controls

        # -------------------------------------------------------------------------
        # Set up group-boxes:
        # -------------------------------------------------------------------------

        # - Project controls
        self.project_box_layout = QtWidgets.QVBoxLayout()
        self.project_box_layout.addLayout(btn_project_layout)
        self.project_box_layout.addWidget(self.lbl_filename)
        self.project_box_layout.addWidget(self.lbl_location)
        self.project_box_layout.addLayout(self.btn_set_model_layout)
        self.project_box = GUI_custom_components.GroupBox('Active project', menu_action=self.ui_obj.menu.toggle_project_control_action)
        self.project_box.setLayout(self.project_box_layout)

        # - Image controls
        self.image_box_layout = QtWidgets.QVBoxLayout()
        self.image_box_layout.addLayout(btn_image_btns_layout)
        self.image_box_layout.addWidget(self.chb_lock_views)
        self.image_box_layout.addWidget(self.lbl_image_width)
        self.image_box_layout.addWidget(self.lbl_image_height)
        self.image_box = GUI_custom_components.GroupBox('Image', menu_action=self.ui_obj.menu.toggle_image_control_action)
        self.image_box.setLayout(self.image_box_layout)

        # - Column detection controls
        self.alg_1_box_layout = QtWidgets.QVBoxLayout()
        self.alg_1_box_layout.addLayout(btn_alg_1_btns_layout)
        self.alg_1_box_layout.addWidget(self.lbl_pixel_average)
        self.alg_1_box_layout.addWidget(self.lbl_search_matrix_peak)
        self.alg_1_box_layout.addWidget(self.lbl_atomic_radii)
        self.alg_1_box_layout.addWidget(self.lbl_overhead_radii)
        self.alg_1_box_layout.addWidget(self.lbl_num_detected_columns)
        self.alg_1_box_layout.addLayout(self.btn_set_scale_layout)
        self.alg_1_box_layout.addLayout(self.btn_set_threshold_layout)
        self.alg_1_box_layout.addLayout(self.btn_set_search_size_layout)
        self.alg_1_box_layout.addWidget(self.chb_automatic_threshold)
        self.alg_1_box_layout.addWidget(self.chb_plot_results)
        self.alg_1_box_layout.addWidget(self.chb_toggle_positions)
        self.alg_1_box = GUI_custom_components.GroupBox('Column detection', menu_action=self.ui_obj.menu.toggle_alg_1_control_action)
        self.alg_1_box.setLayout(self.alg_1_box_layout)

        # - Column characterization controls
        self.alg_2_box_layout = QtWidgets.QVBoxLayout()
        self.alg_2_box_layout.addLayout(btn_alg_2_btns_layout)
        self.alg_2_box_layout.addLayout(self.btn_set_alloy_layout)
        self.alg_2_box_layout.addWidget(self.chb_show_graphic_updates)
        self.alg_2_box = GUI_custom_components.GroupBox('Column characterization', menu_action=self.ui_obj.menu.toggle_alg_2_control_action)
        self.alg_2_box.setLayout(self.alg_2_box_layout)

        # - selected column controls
        self.column_box_layout = QtWidgets.QVBoxLayout()
        self.column_box_layout.addLayout(btn_column_btns_layout)
        self.column_box_layout.addLayout(self.btn_find_column_layout)
        self.column_box_layout.addLayout(self.btn_set_level_layout)
        self.column_box_layout.addLayout(self.btn_set_advanced_species_layout)
        self.column_box_layout.addWidget(self.lbl_column_species)
        self.column_box_layout.addWidget(self.lbl_column_x_pos)
        self.column_box_layout.addWidget(self.lbl_column_y_pos)
        self.column_box_layout.addWidget(self.lbl_prob_vector)
        self.column_box_layout.addLayout(self.probGraphicLayout)
        self.column_box_layout.addWidget(self.chb_precipitate_column)
        self.column_box_layout.addWidget(self.chb_show)
        self.column_box_layout.addWidget(self.lbl_column_peak_gamma)
        self.column_box_layout.addWidget(self.lbl_column_avg_gamma)
        self.column_box_layout.addWidget(self.lbl_norm_peak_gamma)
        self.column_box_layout.addWidget(self.lbl_norm_avg_gamma)
        self.column_box_layout.addWidget(self.lbl_alpha_max)
        self.column_box_layout.addWidget(self.lbl_alpha_min)
        self.column_box_layout.addWidget(self.lbl_theta_max)
        self.column_box_layout.addWidget(self.lbl_theta_min)
        self.column_box_layout.addWidget(self.lbl_theta_mean)
        self.column_box_layout.addWidget(self.lbl_central_variance)
        self.column_box_layout.addLayout(btn_move_control_layout)
        self.column_box = GUI_custom_components.GroupBox('Selected column', menu_action=self.ui_obj.menu.toggle_column_control_action)
        self.column_box.setLayout(self.column_box_layout)

        # - Atomic graph controls
        self.graph_box_layout = QtWidgets.QVBoxLayout()
        self.graph_box_layout.addLayout(btn_graph_btns_layout)
        self.graph_box_layout.addWidget(self.chb_perturb_mode)
        self.graph_box_layout.addWidget(self.chb_enable_ruler)
        self.graph_box_layout.addWidget(self.chb_graph)
        self.graph_box_layout.addWidget(self.chb_toggle_mesh)
        self.graph_box_layout.addWidget(self.lbl_chi)
        self.graph_box = GUI_custom_components.GroupBox('Atomic graph', menu_action=self.ui_obj.menu.toggle_graph_control_action)
        self.graph_box.setLayout(self.graph_box_layout)

        self.sub_graphs_box_layout = QtWidgets.QVBoxLayout()
        self.sub_graphs_box_layout.addLayout(btn_sub_graphs_layout)
        self.sub_graphs_box_layout.addLayout(self.btn_set_sub_graph_type_layout)
        self.sub_graphs_box_layout.addLayout(self.btn_set_sub_graph_order_layout)
        self.sub_graphs_box = GUI_custom_components.GroupBox('Sub-graphs', menu_action=self.ui_obj.menu.toggle_sub_graphs_control_action)
        self.sub_graphs_box.setLayout(self.sub_graphs_box_layout)

        self.anti_graph_box_layout = QtWidgets.QVBoxLayout()
        self.anti_graph_box_layout.addLayout(btn_anti_graph_layout)
        self.anti_graph_box_layout.addWidget(self.chb_show_level_0)
        self.anti_graph_box_layout.addWidget(self.chb_show_level_1)
        self.anti_graph_box = GUI_custom_components.GroupBox('Anti-graph', menu_action=self.ui_obj.menu.toggle_anti_graph_control_action)
        self.anti_graph_box.setLayout(self.anti_graph_box_layout)

        self.info_graph_box_layout = QtWidgets.QVBoxLayout()
        self.info_graph_box_layout.addLayout(btn_info_graph_layout)
        self.info_graph_box = GUI_custom_components.GroupBox('Info-graph', menu_action=self.ui_obj.menu.toggle_info_graph_control_action)
        self.info_graph_box.setLayout(self.info_graph_box_layout)

        # - Data and export controls
        self.analysis_box_layout = QtWidgets.QVBoxLayout()
        self.analysis_box_layout.addLayout(btn_analysis_layout)
        self.analysis_box = GUI_custom_components.GroupBox('Data-analysis', menu_action=self.ui_obj.menu.toggle_analysis_control_action)
        self.analysis_box.setLayout(self.analysis_box_layout)

        # - Overlay controls
        self.overlay_box_layout = QtWidgets.QVBoxLayout()
        self.overlay_box_layout.addLayout(btn_overlay_btns_layout)
        self.overlay_box_layout.addLayout(overlay_layout)
        self.overlay_box = GUI_custom_components.GroupBox('Overlay settings', menu_action=self.ui_obj.menu.toggle_overlay_control_action)
        self.overlay_box.setLayout(self.overlay_box_layout)

        # - Debug controls
        self.debug_box_layout = QtWidgets.QVBoxLayout()
        self.debug_box_layout.addLayout(btn_debug_btns_layout)
        self.debug_box_layout.addLayout(self.btn_set_start_layout)
        self.debug_box = GUI_custom_components.GroupBox('Advanced debug mode',menu_action=self.ui_obj.menu.advanced_debug_mode_action)
        self.debug_box.setLayout(self.debug_box_layout)

        # -------------------------------------------------------------------------
        # Set up top-level layout:
        # -------------------------------------------------------------------------
        self.info_display_layout = QtWidgets.QVBoxLayout()
        self.info_display_layout.addWidget(self.project_box)
        self.info_display_layout.addWidget(self.image_box)
        self.info_display_layout.addWidget(self.alg_1_box)
        self.info_display_layout.addWidget(self.alg_2_box)
        self.info_display_layout.addWidget(self.column_box)
        self.info_display_layout.addWidget(self.graph_box)
        self.info_display_layout.addWidget(self.sub_graphs_box)
        self.info_display_layout.addWidget(self.anti_graph_box)
        self.info_display_layout.addWidget(self.info_graph_box)
        self.info_display_layout.addWidget(self.analysis_box)
        self.info_display_layout.addWidget(self.overlay_box)
        self.info_display_layout.addWidget(self.debug_box)
        self.info_display_layout.addStretch()

        self.setLayout(self.info_display_layout)

        # Make button and checkbox lists to enable looping over all widgets
        self.set_btn_list = []
        self.btn_move_list = []
        self.btn_list = []
        self.chb_list = []

        self.set_btn_list.append(self.btn_set_model_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_threshold_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_search_size_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_scale_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_alloy_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_start_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_find_column_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_level_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_sub_graph_type_layout.itemAt(0).widget())
        self.set_btn_list.append(self.btn_set_sub_graph_order_layout.itemAt(0).widget())

        self.btn_move_list.append(self.btn_cancel_move)
        self.btn_move_list.append(self.btn_set_move)

        self.btn_list.append(self.btn_new_project)
        self.btn_list.append(self.btn_open_project)
        self.btn_list.append(self.btn_save_project)
        self.btn_list.append(self.btn_close_project)
        self.btn_list.append(self.btn_configure_classes)
        self.btn_list.append(self.btn_show_stats)
        self.btn_list.append(self.btn_show_source)
        self.btn_list.append(self.btn_align_views)
        self.btn_list.append(self.btn_export)
        self.btn_list.append(self.btn_start_alg_1)
        self.btn_list.append(self.btn_reset_alg_1)
        self.btn_list.append(self.btn_redraw_search_mat)
        self.btn_list.append(self.btn_start_alg_2)
        self.btn_list.append(self.btn_reset_alg_2)
        self.btn_list.append(self.btn_invert_lvl_alg_2)
        self.btn_list.append(self.btn_set_variant)
        self.btn_list.append(self.btn_delete)
        self.btn_list.append(self.btn_print_details)
        self.btn_list.append(self.btn_snap)
        self.btn_list.append(self.btn_sub)
        self.btn_list.append(self.btn_refresh_graph)
        self.btn_list.append(self.btn_refresh_mesh)
        self.btn_list.append(self.btn_deselect)
        self.btn_list.append(self.btn_new)
        self.btn_list.append(self.btn_set_style)
        self.btn_list.append(self.btn_show_all)
        self.btn_list.append(self.btn_hide_all)
        self.btn_list.append(self.btn_test)
        self.btn_list.append(self.btn_crash)
        self.btn_list.append(self.btn_plot)
        self.btn_list.append(self.btn_print_distances)
        self.btn_list.append(self.btn_build_anti_graph)
        self.btn_list.append(self.btn_build_info_graph)
        self.btn_list.append(self.btn_pca)
        self.btn_list.append(self.btn_calc_models)
        self.btn_list.append(self.btn_plot_models)

        self.chb_list.append(self.chb_lock_views)
        self.chb_list.append(self.chb_plot_results)
        self.chb_list.append(self.chb_automatic_threshold)
        self.chb_list.append(self.chb_toggle_positions)
        self.chb_list.append(self.chb_show_graphic_updates)
        self.chb_list.append(self.chb_precipitate_column)
        self.chb_list.append(self.chb_show)
        self.chb_list.append(self.chb_move)
        self.chb_list.append(self.chb_perturb_mode)
        self.chb_list.append(self.chb_enable_ruler)
        self.chb_list.append(self.chb_graph)
        self.chb_list.append(self.chb_toggle_mesh)
        self.chb_list.append(self.chb_show_level_0)
        self.chb_list.append(self.chb_show_level_1)
        self.chb_list.append(self.chb_raw_image)
        self.chb_list.append(self.chb_black_background)
        self.chb_list.append(self.chb_structures)
        self.chb_list.append(self.chb_si_columns)
        self.chb_list.append(self.chb_si_network)
        self.chb_list.append(self.chb_mg_columns)
        self.chb_list.append(self.chb_particle)
        self.chb_list.append(self.chb_al_columns)
        self.chb_list.append(self.chb_al_network)
        self.chb_list.append(self.chb_cu_columns)
        self.chb_list.append(self.chb_cu_network)
        self.chb_list.append(self.chb_ag_columns)
        self.chb_list.append(self.chb_ag_network)
        self.chb_list.append(self.chb_un_columns)
        self.chb_list.append(self.chb_columns)
        self.chb_list.append(self.chb_al_mesh)
        self.chb_list.append(self.chb_neighbours)
        self.chb_list.append(self.chb_legend)
        self.chb_list.append(self.chb_scalebar)
        self.chb_list.append(self.chb_0_plane)
        self.chb_list.append(self.chb_1_plane)

        # Set tooltips:
        self.mode_tooltip(self.ui_obj.menu.toggle_tooltips_action.isChecked())

    def mode_tooltip(self, on):
        if on:
            for widget, tooltip in zip(self.set_btn_list, GUI_tooltips.control_window_set_list):
                widget.setToolTip(tooltip)
            for widget, tooltip in zip(self.btn_move_list, GUI_tooltips.control_window_move_list):
                widget.setToolTip(tooltip)
            for widget, tooltip in zip(self.btn_list, GUI_tooltips.control_window_btn_list):
                widget.setToolTip(tooltip)
            for widget, tooltip in zip(self.chb_list, GUI_tooltips.control_window_chb_list):
                widget.setToolTip(tooltip)
        else:
            for widget in self.set_btn_list:
                widget.setToolTip('')
            for widget in self.btn_move_list:
                widget.setToolTip('')
            for widget in self.btn_list:
                widget.setToolTip('')
            for widget in self.chb_list:
                widget.setToolTip('')

    def mode_move(self, on):
        if self.ui_obj.project_instance is not None and not self.ui_obj.selected_column == -1:
            self.chb_move.blockSignals(True)
            self.chb_move.setChecked(on)
            self.chb_move.blockSignals(False)
            for i, pos_obj in enumerate(self.ui_obj.gs_atomic_positions.interactive_position_objects):
                if not i == self.ui_obj.selected_column:
                    pos_obj.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, not on)
                else:
                    pos_obj.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, on)
                self.ui_obj.gs_overlay_composition.interactive_overlay_objects[i].setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, not on)
                self.ui_obj.gs_atomic_graph.interactive_vertex_objects[i].setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, not on)
            for chb in self.chb_list:
                chb.setDisabled(on)
            for btn in self.btn_list:
                btn.setDisabled(on)
            for btn in self.set_btn_list:
                btn.setDisabled(on)
            self.btn_set_move.setDisabled(not on)
            self.btn_cancel_move.setDisabled(not on)

    def draw_histogram(self):

        box_width = 15
        box_seperation = 10
        box_displacement = 25

        species = set()
        colors = {}
        if self.ui_obj.project_instance is None:
            species.add('Si')
            colors['Si'] = (255, 0, 0)
            species.add('Cu')
            colors['Cu'] = (255, 255, 0)
            species.add('Al')
            colors['Al'] = (0, 255, 0)
            species.add('Mg')
            colors['Mg'] = (138, 43, 226)
            species.add('Un')
            colors['Un'] = (0, 0, 255)
        else:
            for item in self.ui_obj.project_instance.graph.species_dict.values():
                species.add(item['atomic_species'])
                colors[item['atomic_species']] = item['species_color']

        species = list(species)
        box_heights = []
        if not self.ui_obj.selected_column == -1:
            for atomic_species in species:
                box_heights.append(int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].probability_vector[atomic_species]))
        else:
            box_heights = [0] * 5

        box = QtWidgets.QGraphicsRectItem(0, -10, self.width - 10, self.height - 10)
        box.setPen(GUI_settings.pen_boarder)
        box.hide()

        probGraphicScene = QtWidgets.QGraphicsScene()
        probGraphicScene.addItem(box)

        for i, box_height in enumerate(box_heights):
            x_box = box_seperation + i * box_displacement
            species_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
            species_box.setBrush(QtGui.QBrush(QtGui.QColor(
                colors[species[i]][0],
                colors[species[i]][1],
                colors[species[i]][2]
            )))
            species_box.setX(x_box)
            species_box.setY(100 - box_height)
            species_text = QtWidgets.QGraphicsSimpleTextItem()
            species_text.setText(species[i])
            species_text.setFont(GUI_settings.font_tiny)
            species_text.setX(x_box + 2)
            species_text.setY(100 + 4)
            species_number = QtWidgets.QGraphicsSimpleTextItem()
            species_number.setText(str(box_height / 100))
            species_number.setFont(GUI_settings.font_tiny)
            species_number.setX(x_box - 1)
            species_number.setY(100 - box_height - 10)

            probGraphicScene.addItem(species_box)
            probGraphicScene.addItem(species_text)
            probGraphicScene.addItem(species_number)

        if GUI_settings.theme == 'dark':
            probGraphicScene.palette().setColor(QtGui.QPalette.Text, QtCore.Qt.white)

        self.probGraphicView.setScene(probGraphicScene)

    def select_column(self):

        i = self.ui_obj.selected_column

        if i == -1:
            self.deselect_column()
        else:

            vertex = self.ui_obj.project_instance.graph.vertices[i]

            self.lbl_column_index.setText('Column index: {}'.format(i))
            self.lbl_column_x_pos.setText('x: {:.3f}'.format(vertex.im_coor_x))
            self.lbl_column_y_pos.setText('y: {:.3f}'.format(vertex.im_coor_y))
            self.lbl_column_peak_gamma.setText('Peak gamma: {:.3f}'.format(vertex.peak_gamma))
            self.lbl_column_avg_gamma.setText('Avg gamma: {:.3f}'.format(vertex.avg_gamma))
            self.lbl_column_species.setText('Atomic species: {}'.format(vertex.atomic_species))
            self.lbl_advanced_species.setText('Advanced species: {}'.format(vertex.advanced_species))
            self.lbl_column_level.setText('Zeta: {}'.format(vertex.zeta))
            self.lbl_alpha_max.setText('Alpha max: {:.3f}'.format(vertex.alpha_max))
            self.lbl_alpha_min.setText('Alpha min: {:.3f}'.format(vertex.alpha_min))
            self.lbl_theta_max.setText('Theta max: {:.3f}'.format(vertex.theta_max))
            self.lbl_theta_min.setText('Theta min: {:.3f}'.format(vertex.theta_min))
            self.lbl_central_variance.setText('Theta variance: {:.3f}'.format(vertex.theta_angle_variance))
            self.lbl_theta_mean.setText('Theta mean: {:.3f}'.format(vertex.theta_angle_mean))
            self.lbl_norm_peak_gamma.setText('Normalized peak gamma: {:.3f}'.format(vertex.normalized_peak_gamma))
            self.lbl_norm_avg_gamma.setText('Normalized avg gamma: {:.3f}'.format(vertex.normalized_avg_gamma))

            self.btn_new.setDisabled(False)
            self.btn_deselect.setDisabled(False)
            self.btn_delete.setDisabled(False)
            self.btn_set_level_layout.itemAt(0).widget().setDisabled(False)
            self.btn_find_column_layout.itemAt(0).widget().setDisabled(False)
            self.chb_precipitate_column.setDisabled(False)
            self.chb_show.setDisabled(False)
            self.chb_move.setDisabled(False)

            self.chb_show.blockSignals(True)
            self.chb_show.setChecked(self.ui_obj.project_instance.graph.vertices[i].show_in_overlay)
            self.chb_show.blockSignals(False)
            self.chb_precipitate_column.blockSignals(True)
            self.chb_precipitate_column.setChecked(self.ui_obj.project_instance.graph.vertices[i].is_in_precipitate)
            self.chb_precipitate_column.blockSignals(False)
            self.chb_move.blockSignals(True)
            self.chb_move.setChecked(False)
            self.chb_move.blockSignals(False)

            self.btn_set_move.setDisabled(True)
            self.btn_cancel_move.setDisabled(True)

            self.draw_histogram()

    def deselect_column(self):

        self.draw_histogram()

        self.lbl_column_index.setText('Column index: ')
        self.lbl_column_x_pos.setText('x: ')
        self.lbl_column_y_pos.setText('y: ')
        self.lbl_column_peak_gamma.setText('Peak gamma: ')
        self.lbl_column_avg_gamma.setText('Avg gamma: ')
        self.lbl_column_species.setText('Atomic species: ')
        self.lbl_advanced_species.setText('Advanced species: ')
        self.lbl_column_level.setText('Zeta: ')
        self.lbl_alpha_max.setText('Alpha max: ')
        self.lbl_alpha_min.setText('Alpha min: ')
        self.lbl_theta_max.setText('Theta max: ')
        self.lbl_theta_min.setText('Theta min: ')
        self.lbl_central_variance.setText('Theta variance: ')
        self.lbl_theta_mean.setText('Theta mean: ')
        self.lbl_norm_peak_gamma.setText('Normalized peak gamma: ')
        self.lbl_norm_avg_gamma.setText('Normalized avg gamma: ')

    def update_display(self):

        if self.ui_obj.project_instance is not None:

            self.lbl_model.setText('Associated model: {}'.format(self.ui_obj.project_instance.graph.active_model))
            if self.ui_obj.savefile:
                self.lbl_filename.setText('Filename: {}'.format(os.path.split(self.ui_obj.savefile)[1]))
                self.lbl_location.setText('Location: {}'.format(os.path.split(self.ui_obj.savefile)[0]))
            else:
                self.lbl_filename.setText('Filename: ')
                self.lbl_location.setText('Location: ')

            self.lbl_num_detected_columns.setText('Number of detected columns: {}'.format(self.ui_obj.project_instance.num_columns))
            self.lbl_image_width.setText('Image width (pixels): {}'.format(self.ui_obj.project_instance.im_width))
            self.lbl_image_height.setText('Image height (pixels): {}'.format(self.ui_obj.project_instance.im_height))

            self.lbl_starting_index.setText('Default starting index: {}'.format(self.ui_obj.project_instance.starting_index))

            self.lbl_atomic_radii.setText('Approx atomic radii (pixels): {}'.format(self.ui_obj.project_instance.r))
            self.lbl_overhead_radii.setText('Overhead (pixels): {}'.format(self.ui_obj.project_instance.overhead))
            self.lbl_detection_threshold.setText('Detection threshold value: {}'.format(self.ui_obj.project_instance.threshold))
            self.lbl_search_matrix_peak.setText('Search matrix peak: {:.3f}'.format(self.ui_obj.project_instance.search_mat.max()))
            self.lbl_search_size.setText('Search size: {}'.format(self.ui_obj.project_instance.search_size))
            self.lbl_scale.setText('Scale (pm / pixel): {:8f}'.format(self.ui_obj.project_instance.scale))
            self.lbl_pixel_average.setText('Average pixel value: {:.3f}'.format(self.ui_obj.project_instance.pixel_average))

            self.lbl_alloy.setText('Alloy: {}'.format(self.ui_obj.project_instance.get_alloy_string()))

            if self.ui_obj.selected_column == -1:
                self.deselect_column()
            else:
                self.select_column()

            self.lbl_chi.setText('Chi: {:.3f}'.format(self.ui_obj.project_instance.graph.chi))

        else:

            self.empty_display()

    def empty_display(self):

        self.deselect_column()

        self.lbl_model.setText('Associated model: ')
        self.lbl_filename.setText('Filename: ')
        self.lbl_location.setText('Location: ')

        self.lbl_num_detected_columns.setText('Number of detected columns: ')
        self.lbl_image_width.setText('Image width (pixels): ')
        self.lbl_image_height.setText('Image height (pixels): ')

        self.lbl_atomic_radii.setText('Approx atomic radii (pixels): ')
        self.lbl_overhead_radii.setText('Overhead (pixels): ')
        self.lbl_detection_threshold.setText('Detection threshold value: ')
        self.lbl_search_matrix_peak.setText('Search matrix peak: ')
        self.lbl_search_size.setText('Search size: ')
        self.lbl_scale.setText('Scale (pm / pixel): ')
        self.lbl_pixel_average.setText('Average pixel value: ')

        self.lbl_alloy.setText('Alloy: ')

        self.lbl_starting_index.setText('Default starting index: ')

        self.lbl_chi.setText('Chi: ')

    def keyPressEvent(self, event):

        self.ui_obj.keyPressEvent(event)


class MenuBar:

    def __init__(self, bar_obj, ui_obj):

        self.bar_obj = bar_obj
        self.ui_obj = ui_obj

        # Increase the visibility of separators when 'dark' theme!
        self.bar_obj.setStyleSheet("""
                            QMenu::separator {
                                height: 1px;
                                background: grey;
                                margin-left: 10px;
                                margin-right: 5px;
                            }
                        """)

        # Main headers
        file = self.bar_obj.addMenu('File')
        edit = self.bar_obj.addMenu('Edit')
        view = self.bar_obj.addMenu('View')
        process = self.bar_obj.addMenu('Process')
        column_detection = process.addMenu('Column detection')
        column_analysis = process.addMenu('Column analysis')
        export = self.bar_obj.addMenu('Export')
        image = export.addMenu('Image')
        debug = self.bar_obj.addMenu('Debug')
        plugins_ = self.bar_obj.addMenu('Plugins')
        help = self.bar_obj.addMenu('Help')

        # Create actions for menus
        # - file
        new_action = QtWidgets.QAction('New', self.ui_obj)
        open_action = QtWidgets.QAction('Open', self.ui_obj)
        save_action = QtWidgets.QAction('Save', self.ui_obj)
        close_action = QtWidgets.QAction('Close', self.ui_obj)
        exit_action = QtWidgets.QAction('Exit', self.ui_obj)
        # - edit
        # - view
        view_image_title_action = QtWidgets.QAction('View path of original image', self.ui_obj)
        show_stats_action = QtWidgets.QAction('Show image statistics', self.ui_obj)
        update_display_action = QtWidgets.QAction('Update display', self.ui_obj)
        self.toggle_project_control_action = QtWidgets.QAction('Show project controls', self.ui_obj)
        self.toggle_project_control_action.setCheckable(True)
        self.toggle_project_control_action.setChecked(True)
        self.toggle_image_control_action = QtWidgets.QAction('Show image controls', self.ui_obj)
        self.toggle_image_control_action.setCheckable(True)
        self.toggle_image_control_action.setChecked(True)
        self.toggle_alg_1_control_action = QtWidgets.QAction('Show column detection controls', self.ui_obj)
        self.toggle_alg_1_control_action.setCheckable(True)
        self.toggle_alg_1_control_action.setChecked(True)
        self.toggle_alg_2_control_action = QtWidgets.QAction('Show column characterization controls', self.ui_obj)
        self.toggle_alg_2_control_action.setCheckable(True)
        self.toggle_alg_2_control_action.setChecked(True)
        self.toggle_column_control_action = QtWidgets.QAction('Show selected column controls', self.ui_obj)
        self.toggle_column_control_action.setCheckable(True)
        self.toggle_column_control_action.setChecked(True)
        self.toggle_graph_control_action = QtWidgets.QAction('Show atomic graph controls', self.ui_obj)
        self.toggle_graph_control_action.setCheckable(True)
        self.toggle_graph_control_action.setChecked(True)
        self.toggle_sub_graphs_control_action = QtWidgets.QAction('Show sub-graph controls', self.ui_obj)
        self.toggle_sub_graphs_control_action.setCheckable(True)
        self.toggle_sub_graphs_control_action.setChecked(True)
        self.toggle_anti_graph_control_action = QtWidgets.QAction('Show anti-graph controls', self.ui_obj)
        self.toggle_anti_graph_control_action.setCheckable(True)
        self.toggle_anti_graph_control_action.setChecked(True)
        self.toggle_info_graph_control_action = QtWidgets.QAction('Show info-graph controls', self.ui_obj)
        self.toggle_info_graph_control_action.setCheckable(True)
        self.toggle_info_graph_control_action.setChecked(True)
        self.toggle_overlay_control_action = QtWidgets.QAction('Show overlay controls', self.ui_obj)
        self.toggle_overlay_control_action.setCheckable(True)
        self.toggle_overlay_control_action.setChecked(True)
        self.toggle_analysis_control_action = QtWidgets.QAction('Show analysis controls', self.ui_obj)
        self.toggle_analysis_control_action.setCheckable(True)
        self.toggle_analysis_control_action.setChecked(True)
        # - Process
        image_correction_action = QtWidgets.QAction('Image corrections', self.ui_obj)
        image_filter_action = QtWidgets.QAction('Image filters', self.ui_obj)
        image_adjustments_action = QtWidgets.QAction('Image adjustments', self.ui_obj)
        continue_detection_action = QtWidgets.QAction('Continue column detection', self.ui_obj)
        restart_detection_action = QtWidgets.QAction('Restart column detection', self.ui_obj)
        continue_analysis_action = QtWidgets.QAction('Continue column analysis', self.ui_obj)
        restart_analysis_action = QtWidgets.QAction('Restart column analysis', self.ui_obj)
        # - export
        export_data_action = QtWidgets.QAction('Export data', self.ui_obj)
        export_raw_image_action = QtWidgets.QAction('Export raw image', self.ui_obj)
        export_column_position_image_action = QtWidgets.QAction('Export column position image', self.ui_obj)
        export_overlay_image_action = QtWidgets.QAction('Export overlay image', self.ui_obj)
        export_atomic_graph_action = QtWidgets.QAction('Export atomic graph', self.ui_obj)
        export_zeta_graph_action = QtWidgets.QAction('Export zeta graph', self.ui_obj)
        make_plots_action = QtWidgets.QAction('Make plots', self.ui_obj)
        # - Debug
        self.advanced_debug_mode_action = QtWidgets.QAction('Advanced debug mode', self.ui_obj)
        self.advanced_debug_mode_action.setCheckable(True)
        self.advanced_debug_mode_action.blockSignals(True)
        self.advanced_debug_mode_action.setChecked(False)
        self.advanced_debug_mode_action.blockSignals(False)
        add_mark_action = QtWidgets.QAction('Add mark to terminal', self.ui_obj)
        reset_flags_action = QtWidgets.QAction('Reset all flags', self.ui_obj)
        set_control_file_action = QtWidgets.QAction('Set control instance', self.ui_obj)
        display_deviations_action = QtWidgets.QAction('Display deviation stats', self.ui_obj)
        run_validation_test_action = QtWidgets.QAction('Run algorithm benchmark', self.ui_obj)
        test_consistency_action = QtWidgets.QAction('Reset levels', self.ui_obj)
        invert_precipitate_levels_action = QtWidgets.QAction('Invert precipitate levels', self.ui_obj)
        ad_hoc_action = QtWidgets.QAction('Ad Hoc functionality', self.ui_obj)
        # - Plugins
        self.plugin_actions = []
        plugin_paths = []
        for plugin in pathlib.Path('plugins/').glob('*.py'):
            plugin_paths.append(plugin)
            self.plugin_actions.append(QtWidgets.QAction(os.path.splitext(plugin.name)[0], self.ui_obj))
        # - help
        self.toggle_tooltips_action = QtWidgets.QAction('Show tooltips', self.ui_obj)
        self.toggle_tooltips_action.setCheckable(True)
        self.toggle_tooltips_action.setChecked(GUI_settings.tooltips)
        set_theme_action = QtWidgets.QAction('Set theme', self.ui_obj)
        there_is_no_help_action = QtWidgets.QAction('HJALP!', self.ui_obj)

        # Add actions to menus
        # - file
        file.addAction(new_action)
        file.addAction(open_action)
        file.addAction(save_action)
        file.addAction(close_action)
        file.addAction(exit_action)
        # - edit
        # - View
        view.addAction(view_image_title_action)
        view.addAction(show_stats_action)
        view.addAction(update_display_action)
        view.addSeparator()
        view.addAction(self.toggle_project_control_action)
        view.addAction(self.toggle_image_control_action)
        view.addAction(self.toggle_alg_1_control_action)
        view.addAction(self.toggle_alg_2_control_action)
        view.addAction(self.toggle_column_control_action)
        view.addAction(self.toggle_graph_control_action)
        view.addAction(self.toggle_sub_graphs_control_action)
        view.addAction(self.toggle_anti_graph_control_action)
        view.addAction(self.toggle_info_graph_control_action)
        view.addAction(self.toggle_overlay_control_action)
        view.addAction(self.toggle_analysis_control_action)
        # - Process
        process.addAction(image_correction_action)
        process.addAction(image_filter_action)
        process.addAction(image_adjustments_action)
        column_detection.addAction(continue_detection_action)
        column_detection.addAction(restart_detection_action)
        column_analysis.addAction(continue_analysis_action)
        column_analysis.addAction(restart_analysis_action)
        # - Export
        export.addAction(export_data_action)
        export.addAction(make_plots_action)
        image.addAction(export_raw_image_action)
        image.addAction(export_column_position_image_action)
        image.addAction(export_overlay_image_action)
        image.addAction(export_atomic_graph_action)
        image.addAction(export_zeta_graph_action)
        # - Debug
        debug.addAction(self.advanced_debug_mode_action)
        debug.addAction(add_mark_action)
        debug.addAction(reset_flags_action)
        debug.addAction(set_control_file_action)
        debug.addAction(run_validation_test_action)
        debug.addAction(display_deviations_action)
        debug.addAction(test_consistency_action)
        debug.addAction(invert_precipitate_levels_action)
        debug.addAction(ad_hoc_action)
        # - Plugins
        for plugin in self.plugin_actions:
            plugins_.addAction(plugin)
        # - Help
        help.addAction(self.toggle_tooltips_action)
        help.addSeparator()
        help.addAction(set_theme_action)
        help.addAction(there_is_no_help_action)

        # Events
        # - file
        new_action.triggered.connect(self.ui_obj.menu_new_trigger)
        open_action.triggered.connect(self.ui_obj.menu_open_trigger)
        save_action.triggered.connect(self.ui_obj.menu_save_trigger)
        close_action.triggered.connect(self.ui_obj.menu_close_trigger)
        exit_action.triggered.connect(self.ui_obj.menu_exit_trigger)
        # - edit
        # - view
        view_image_title_action.triggered.connect(self.ui_obj.menu_view_image_title_trigger)
        show_stats_action.triggered.connect(self.ui_obj.menu_show_stats_trigger)
        update_display_action.triggered.connect(self.ui_obj.menu_update_display)
        self.toggle_project_control_action.triggered.connect(self.ui_obj.menu_toggle_project_control_trigger)
        self.toggle_image_control_action.triggered.connect(self.ui_obj.menu_toggle_image_control_trigger)
        self.toggle_alg_1_control_action.triggered.connect(self.ui_obj.menu_toggle_alg_1_control_trigger)
        self.toggle_alg_2_control_action.triggered.connect(self.ui_obj.menu_toggle_alg_2_control_trigger)
        self.toggle_column_control_action.triggered.connect(self.ui_obj.menu_toggle_column_control_trigger)
        self.toggle_graph_control_action.triggered.connect(self.ui_obj.menu_toggle_graph_control_trigger)
        self.toggle_sub_graphs_control_action.triggered.connect(self.ui_obj.menu_toggle_sub_graphs_control_trigger)
        self.toggle_anti_graph_control_action.triggered.connect(self.ui_obj.menu_toggle_anti_graph_control_trigger)
        self.toggle_info_graph_control_action.triggered.connect(self.ui_obj.menu_toggle_info_graph_control_trigger)
        self.toggle_overlay_control_action.triggered.connect(self.ui_obj.menu_toggle_overlay_control_trigger)
        self.toggle_analysis_control_action.triggered.connect(self.ui_obj.menu_toggle_analysis_control_trigger)
        # - Process
        image_correction_action.triggered.connect(self.ui_obj.menu_image_correction_trigger)
        image_filter_action.triggered.connect(self.ui_obj.menu_image_filter_trigger)
        image_adjustments_action.triggered.connect(self.ui_obj.menu_image_adjustments_trigger)
        continue_detection_action.triggered.connect(self.ui_obj.menu_continue_detection_trigger)
        restart_detection_action.triggered.connect(self.ui_obj.menu_restart_detection_trigger)
        continue_analysis_action.triggered.connect(self.ui_obj.menu_continue_analysis_trigger)
        restart_analysis_action.triggered.connect(self.ui_obj.menu_restart_analysis_trigger)
        # - Export
        export_data_action.triggered.connect(self.ui_obj.menu_export_data_trigger)
        make_plots_action.triggered.connect(self.ui_obj.menu_make_plots_trigger)
        export_raw_image_action.triggered.connect(self.ui_obj.menu_export_raw_image_trigger)
        export_column_position_image_action.triggered.connect(self.ui_obj.menu_export_column_position_image_trigger)
        export_overlay_image_action.triggered.connect(self.ui_obj.menu_export_overlay_image_trigger)
        export_atomic_graph_action.triggered.connect(self.ui_obj.menu_export_atomic_graph_trigger)
        export_zeta_graph_action.triggered.connect(self.ui_obj.menu_export_zeta_graph_trigger)
        # - debug
        self.advanced_debug_mode_action.triggered.connect(self.ui_obj.menu_toggle_debug_mode_trigger)
        add_mark_action.triggered.connect(self.ui_obj.menu_add_mark_trigger)
        reset_flags_action.triggered.connect(self.ui_obj.menu_clear_flags_trigger)
        set_control_file_action.triggered.connect(self.ui_obj.menu_set_control_file_trigger)
        run_validation_test_action.triggered.connect(self.ui_obj.menu_run_benchmark_trigger)
        display_deviations_action.triggered.connect(self.ui_obj.menu_display_deviations_trigger)
        test_consistency_action.triggered.connect(self.ui_obj.menu_test_consistency_trigger)
        invert_precipitate_levels_action.triggered.connect(self.ui_obj.menu_invert_precipitate_columns_trigger)
        ad_hoc_action.triggered.connect(self.ui_obj.menu_ad_hoc_trigger)
        # - Plugins (lol, this is so hacky... There must be a better way! - Raymond Hettinger)
        with open('plugin_modules.py', 'w') as imp:
            for path in plugin_paths:
                imp.writelines('import plugins.{}\n'.format(os.path.splitext(path.name)[0]))
        import plugin_modules
        self.plugin_instances = []
        for path, action in zip(plugin_paths, self.plugin_actions):
            module_name = os.path.splitext(path.name)[0]
            plugin_instance = eval('plugin_modules.plugins.{}.Bridge(self.ui_obj)'.format(module_name))
            self.plugin_instances.append(plugin_instance)
            action.triggered.connect(plugin_instance.trigger)
        # - help
        self.toggle_tooltips_action.triggered.connect(self.ui_obj.menu_toggle_tooltips_trigger)
        set_theme_action.triggered.connect(self.ui_obj.menu_set_theme_trigger)
        there_is_no_help_action.triggered.connect(self.ui_obj.menu_there_is_no_help_trigger)


# ----------
# Custom dialogs:
# ----------


class DataExportWizard(QtWidgets.QDialog):

    def __init__(self, *args, ui_obj=None):
        super().__init__(*args)

        self.ui_obj = ui_obj

        self.setWindowTitle('Calculate model parameters wizard')

        self.btn_next = QtWidgets.QPushButton('Next')
        self.btn_next.clicked.connect(self.btn_next_trigger)
        self.btn_back = QtWidgets.QPushButton('Back')
        self.btn_back.clicked.connect(self.btn_back_trigger)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.btn_cancel_trigger)

        self.btn_layout = QtWidgets.QHBoxLayout()
        self.stack_layout = QtWidgets.QStackedLayout()
        self.top_layout = QtWidgets.QVBoxLayout()

        self.widget_frame_0 = QtWidgets.QWidget()
        self.widget_frame_1 = QtWidgets.QWidget()
        self.widget_frame_2 = QtWidgets.QWidget()
        self.widget_frame_3 = QtWidgets.QWidget()
        self.widget_frame_4 = QtWidgets.QWidget()

        # Frame 0
        self.lbl_file = QtWidgets.QLabel('Export data from current project or enter a list of projects')
        self.rbtn_current_project = QtWidgets.QRadioButton('Current project')
        self.rbtn_list_of_projects = QtWidgets.QRadioButton('Enter list of projects files')
        self.lst_files = QtWidgets.QListWidget()
        self.btn_add_files = QtWidgets.QPushButton('Add files')

        # Frame 1
        self.lbl_select_categories = QtWidgets.QLabel('Select the desired categorization')
        self.cmb_categorization = QtWidgets.QComboBox()

        # Frame 2
        self.list_1 = QtWidgets.QListWidget()
        self.list_2 = QtWidgets.QListWidget()
        self.btn_list_1_up = QtWidgets.QPushButton('Move up')
        self.btn_list_1_up.clicked.connect(self.btn_list_1_up_trigger)
        self.btn_list_1_down = QtWidgets.QPushButton('Move down')
        self.btn_list_1_down.clicked.connect(self.btn_list_1_down_trigger)
        self.btn_list_2_up = QtWidgets.QPushButton('Move up')
        self.btn_list_2_up.clicked.connect(self.btn_list_2_up_trigger)
        self.btn_list_2_down = QtWidgets.QPushButton('Move down')
        self.btn_list_2_down.clicked.connect(self.btn_list_2_down_trigger)
        self.btn_add = QtWidgets.QPushButton('Add')
        self.btn_add.clicked.connect(self.btn_add_item_trigger)
        self.btn_remove = QtWidgets.QPushButton('Remove')
        self.btn_remove.clicked.connect(self.btn_remove_item_trigger)
        self.lbl_included_data = QtWidgets.QLabel('Included data-columns:')
        self.lbl_available_data = QtWidgets.QLabel('Available data-columns:')

        # Frame 3
        self.lbl_filter = QtWidgets.QLabel(
            'Set exclusionn filter: (Columns with checked properties will not be included)')
        self.chb_edge_columns = QtWidgets.QCheckBox('Exclude edge columns')
        self.chb_matrix_columns = QtWidgets.QCheckBox('Exclude matrix columns')
        self.chb_particle_columns = QtWidgets.QCheckBox('Exclude particle columns')
        self.chb_hidden_columns = QtWidgets.QCheckBox('Exclude columns that are set as hidden in the overlay')
        self.chb_flag_1 = QtWidgets.QCheckBox('Exclude columns where flag 1 is set to True')
        self.chb_flag_2 = QtWidgets.QCheckBox('Exclude columns where flag 2 is set to True')
        self.chb_flag_3 = QtWidgets.QCheckBox('Exclude columns where flag 3 is set to True')
        self.chb_flag_4 = QtWidgets.QCheckBox('Exclude columns where flag 4 is set to True')

        # Frame 4
        self.chb_recalculate_graphs = QtWidgets.QCheckBox(
            'Recalculate graph data before compiling data (might be very slow)')

        self.step = 0
        self.state_list = []

        self.set_layout()
        self.exec_()

    def page_0_layout(self):
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_file)
        v_layout.addWidget(self.rbtn_current_project)
        v_layout.addWidget(self.rbtn_list_of_projects)
        v_layout.addWidget(self.lst_files)
        v_layout.addWidget(self.btn_add_files)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.rbtn_current_project.setChecked(True)
        self.rbtn_list_of_projects.setChecked(False)
        self.lst_files.setDisabled(True)
        self.btn_add_files.setDisabled(True)

        self.lst_files.setMinimumWidth(500)

        self.rbtn_current_project.toggled.connect(self.rbtn_current_project_trigger)
        self.btn_add_files.clicked.connect(self.btn_add_files_trigger)

        self.widget_frame_0.setLayout(h_layout)

    def page_1_layout(self):
        self.cmb_categorization.addItem('advanced')
        self.cmb_categorization.addItem('simple')
        self.cmb_categorization.addItem('none')

        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_select_categories)
        v_layout.addWidget(self.cmb_categorization)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.widget_frame_1.setLayout(h_layout)

    def page_2_layout(self):
        self.list_2.addItems([
            'i',
            'r',
            'species_index',
            'species_variant',
            'advanced_category_index',
            'n',
            'peak_gamma',
            'avg_gamma',
            'normalized_peak_gamma',
            'normalized_avg_gamma',
            'im_coor_x',
            'im_coor_y',
            'im_coor_z',
            'spatial_coor_x',
            'spatial_coor_y',
            'spatial_coor_z',
            'zeta',
            'alpha_min',
            'alpha_max',
            'theta_min',
            'theta_max',
            'theta_angle_variance',
            'theta_angle_mean',
            'redshift',
            'avg_redshift',
            'avg_central_separation'
        ])

        h_layout = QtWidgets.QHBoxLayout()
        v_layout_1 = QtWidgets.QVBoxLayout()
        v_layout_2 = QtWidgets.QVBoxLayout()
        v_layout_3 = QtWidgets.QVBoxLayout()
        v_layout_4 = QtWidgets.QVBoxLayout()
        v_layout_5 = QtWidgets.QVBoxLayout()

        v_layout_1.addStretch()
        v_layout_1.addWidget(self.btn_list_1_up)
        v_layout_1.addWidget(self.btn_list_1_down)
        v_layout_1.addStretch()

        v_layout_2.addWidget(self.lbl_included_data)
        v_layout_2.addWidget(self.list_1)

        v_layout_3.addStretch()
        v_layout_3.addWidget(self.btn_add)
        v_layout_3.addWidget(self.btn_remove)
        v_layout_3.addStretch()

        v_layout_4.addWidget(self.lbl_available_data)
        v_layout_4.addWidget(self.list_2)

        v_layout_5.addStretch()
        v_layout_5.addWidget(self.btn_list_2_up)
        v_layout_5.addWidget(self.btn_list_2_down)
        v_layout_5.addStretch()

        h_layout.addLayout(v_layout_1)
        h_layout.addLayout(v_layout_2)
        h_layout.addLayout(v_layout_3)
        h_layout.addLayout(v_layout_4)
        h_layout.addLayout(v_layout_5)

        self.widget_frame_2.setLayout(h_layout)

    def page_3_layout(self):
        self.chb_edge_columns.setChecked(True)
        self.chb_matrix_columns.setChecked(False)
        self.chb_particle_columns.setChecked(False)
        self.chb_hidden_columns.setChecked(False)
        self.chb_flag_1.setChecked(False)
        self.chb_flag_2.setChecked(False)
        self.chb_flag_3.setChecked(False)
        self.chb_flag_4.setChecked(False)

        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_filter)
        v_layout.addWidget(self.chb_edge_columns)
        v_layout.addWidget(self.chb_matrix_columns)
        v_layout.addWidget(self.chb_particle_columns)
        v_layout.addWidget(self.chb_hidden_columns)
        v_layout.addWidget(self.chb_flag_1)
        v_layout.addWidget(self.chb_flag_2)
        v_layout.addWidget(self.chb_flag_3)
        v_layout.addWidget(self.chb_flag_4)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.widget_frame_3.setLayout(h_layout)

    def page_4_layout(self):
        self.chb_recalculate_graphs.setChecked(False)
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self.chb_recalculate_graphs)
        v_layout.addStretch()
        self.widget_frame_4.setLayout(v_layout)

    def set_layout(self):
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_cancel)
        self.btn_layout.addWidget(self.btn_back)
        self.btn_layout.addWidget(self.btn_next)
        self.btn_layout.addStretch()

        self.page_0_layout()
        self.stack_layout.addWidget(self.widget_frame_0)

        self.page_1_layout()
        self.stack_layout.addWidget(self.widget_frame_1)

        self.page_2_layout()
        self.stack_layout.addWidget(self.widget_frame_2)

        self.page_3_layout()
        self.stack_layout.addWidget(self.widget_frame_3)

        self.page_4_layout()
        self.stack_layout.addWidget(self.widget_frame_4)

        self.top_layout.addLayout(self.stack_layout)
        self.top_layout.addLayout(self.btn_layout)

        self.setLayout(self.top_layout)

    def btn_add_files_trigger(self):
        prompt = QtWidgets.QFileDialog()
        prompt.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if prompt.exec_():
            filenames = prompt.selectedFiles()
        else:
            filenames = None
        if filenames is not None:
            for file_ in filenames:
                exlusive = True
                for i in range(self.lst_files.count()):
                    if file_ == self.lst_files.item(i).text():
                        exlusive = False
                if exlusive:
                    self.lst_files.addItem(file_)

    def rbtn_current_project_trigger(self, state):
        self.lst_files.setDisabled(state)
        self.btn_add_files.setDisabled(state)

    def btn_list_2_up_trigger(self):
        if self.list_2.currentItem() is not None:
            text = self.list_2.currentItem().text()
            index = self.list_2.currentRow()
            if not index == 0:
                self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))
                self.list_2.insertItem(index - 1, text)
                self.list_2.setCurrentRow(index - 1)

    def btn_list_2_down_trigger(self):
        if self.list_2.currentItem() is not None:
            text = self.list_2.currentItem().text()
            index = self.list_2.currentRow()
            if not index == self.list_2.count() - 1:
                self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))
                self.list_2.insertItem(index + 1, text)
                self.list_2.setCurrentRow(index + 1)

    def btn_list_1_up_trigger(self):
        if self.list_1.currentItem() is not None:
            text = self.list_1.currentItem().text()
            index = self.list_1.currentRow()
            if not index == 0:
                self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))
                self.list_1.insertItem(index - 1, text)
                self.list_1.setCurrentRow(index - 1)

    def btn_list_1_down_trigger(self):
        if self.list_1.currentItem() is not None:
            text = self.list_1.currentItem().text()
            index = self.list_1.currentRow()
            if not index == self.list_1.count() - 1:
                self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))
                self.list_1.insertItem(index + 1, text)
                self.list_1.setCurrentRow(index + 1)

    def btn_add_item_trigger(self):
        if self.list_2.currentItem() is not None:
            self.list_1.addItem(self.list_2.currentItem().text())
            self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))

    def btn_remove_item_trigger(self):
        if self.list_1.currentItem() is not None:
            self.list_2.addItem(self.list_1.currentItem().text())
            self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))

    def calc(self):
        self.close()
        self.ui_obj.sys_message('Working...')

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save model as", '', ".csv")
        if filename[0]:
            # Prepare string with filenames
            if self.rbtn_list_of_projects.isChecked():
                files = ''
                for i in range(self.lst_files.count()):
                    if not i == self.lst_files.count() - 1:
                        files += self.lst_files.item(i).text() + '\n'
                    else:
                        files += self.lst_files.item(i).text()
            else:
                files = self.ui_obj.savefile
            # Prepare filter
            filter_ = {
                'exclude_edge_columns': self.chb_edge_columns.isChecked(),
                'exclude_matrix_columns': self.chb_matrix_columns.isChecked(),
                'exclude_particle_columns': self.chb_particle_columns.isChecked(),
                'exclude_hidden_columns': self.chb_hidden_columns.isChecked(),
                'exclude_flag_1_columns': self.chb_flag_1.isChecked(),
                'exclude_flag_2_columns': self.chb_flag_2.isChecked(),
                'exclude_flag_3_columns': self.chb_flag_3.isChecked(),
                'exclude_flag_4_columns': self.chb_flag_4.isChecked()
            }
            # Prepare keys
            keys = []
            export_keys = []
            for i in range(self.list_1.count()):
                export_keys.append(self.list_1.item(i).text())
            keys = copy.deepcopy(export_keys)
            if self.cmb_categorization.currentText() == 'advanced' and 'advanced_category_index' not in keys:
                keys.append('advanced_category_index')
            if self.cmb_categorization.currentText() == 'simple' and 'species_index' not in export_keys:
                keys.append('species_index')
            # Initate manager
            manager = data_module.VertexDataManager(files, filter_=filter_, save_filename=filename[0], recalc=self.chb_recalculate_graphs.isChecked(), attr_keys=keys)
            manager.export_csv(filename[0] + filename[1], export_keys)
            GUI.logger.info('Successfully exported data to {}'.format(filename[0]))
        self.ui_obj.sys_message('Ready.')

    def get_next_frame(self):
        next_frame = 0
        if self.stack_layout.currentIndex() == 0:
            next_frame = 1
        elif self.stack_layout.currentIndex() == 1:
            next_frame = 2
        elif self.stack_layout.currentIndex() == 2:
            next_frame = 3
        elif self.stack_layout.currentIndex() == 3:
            next_frame = 4
            self.btn_next.setText('Calcuate!')
        elif self.stack_layout.currentIndex() == 4:
            next_frame = -2
        else:
            logger.error('Error!')
            self.close()

        return next_frame

    def get_previous_frame(self):
        previous_frame = 0
        if self.stack_layout.currentIndex() == 0:
            previous_frame = -1
        elif self.stack_layout.currentIndex() == 1:
            previous_frame = 0
        elif self.stack_layout.currentIndex() == 2:
            previous_frame = 1
        elif self.stack_layout.currentIndex() == 3:
            previous_frame = 2
        elif self.stack_layout.currentIndex() == 4:
            previous_frame = 3
        else:
            logger.error('Error!')
            self.close()

        return previous_frame

    def btn_next_trigger(self):
        next_frame = self.get_next_frame()
        if next_frame == -2:
            self.calc()
        elif next_frame == -1:
            self.close()
        else:
            self.stack_layout.setCurrentIndex(next_frame)

    def btn_back_trigger(self):
        previous_frame = self.get_previous_frame()
        if previous_frame == -1:
            self.close()
        else:
            self.stack_layout.setCurrentIndex(previous_frame)

    def btn_cancel_trigger(self):
        self.close()


class CalcModels(QtWidgets.QDialog):

    def __init__(self, *args, ui_obj=None):
        super().__init__(*args)

        self.ui_obj = ui_obj

        self.setWindowTitle('Calculate model parameters wizard')

        self.btn_next = QtWidgets.QPushButton('Next')
        self.btn_next.clicked.connect(self.btn_next_trigger)
        self.btn_back = QtWidgets.QPushButton('Back')
        self.btn_back.clicked.connect(self.btn_back_trigger)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.btn_cancel_trigger)

        self.btn_layout = QtWidgets.QHBoxLayout()
        self.stack_layout = QtWidgets.QStackedLayout()
        self.top_layout = QtWidgets.QVBoxLayout()

        self.widget_frame_0 = QtWidgets.QWidget()
        self.widget_frame_1 = QtWidgets.QWidget()
        self.widget_frame_2 = QtWidgets.QWidget()
        self.widget_frame_3 = QtWidgets.QWidget()
        self.widget_frame_4 = QtWidgets.QWidget()

        # Frame 0
        self.lbl_file = QtWidgets.QLabel('Calc model from current project or enter a list of projects')
        self.rbtn_current_project = QtWidgets.QRadioButton('Current project')
        self.rbtn_list_of_projects = QtWidgets.QRadioButton('Enter list of projects files')
        self.lst_files = QtWidgets.QListWidget()
        self.btn_add_files = QtWidgets.QPushButton('Add files')

        # Frame 1
        self.lbl_nominal_data = QtWidgets.QLabel('Select a nominal data type to categorize by: ')
        self.cmb_nominal_data = QtWidgets.QComboBox()

        # Frame 2
        self.list_1 = QtWidgets.QListWidget()
        self.list_2 = QtWidgets.QListWidget()
        self.btn_list_1_up = QtWidgets.QPushButton('Move up')
        self.btn_list_1_up.clicked.connect(self.btn_list_1_up_trigger)
        self.btn_list_1_down = QtWidgets.QPushButton('Move down')
        self.btn_list_1_down.clicked.connect(self.btn_list_1_down_trigger)
        self.btn_list_2_up = QtWidgets.QPushButton('Move up')
        self.btn_list_2_up.clicked.connect(self.btn_list_2_up_trigger)
        self.btn_list_2_down = QtWidgets.QPushButton('Move down')
        self.btn_list_2_down.clicked.connect(self.btn_list_2_down_trigger)
        self.btn_add = QtWidgets.QPushButton('Add')
        self.btn_add.clicked.connect(self.btn_add_item_trigger)
        self.btn_remove = QtWidgets.QPushButton('Remove')
        self.btn_remove.clicked.connect(self.btn_remove_item_trigger)
        self.lbl_included_data = QtWidgets.QLabel('Included attributes:')
        self.lbl_available_data = QtWidgets.QLabel('Available attributes:')

        # Frame 3
        self.lbl_filter = QtWidgets.QLabel('Set exclusionn filter: (Columns with checked properties will not be included)')
        self.chb_edge_columns = QtWidgets.QCheckBox('Exclude edge columns')
        self.chb_matrix_columns = QtWidgets.QCheckBox('Exclude matrix columns')
        self.chb_particle_columns = QtWidgets.QCheckBox('Exclude particle columns')
        self.chb_hidden_columns = QtWidgets.QCheckBox('Exclude columns that are set as hidden in the overlay')
        self.chb_flag_1 = QtWidgets.QCheckBox('Exclude columns where flag 1 is set to True')
        self.chb_flag_2 = QtWidgets.QCheckBox('Exclude columns where flag 2 is set to True')
        self.chb_flag_3 = QtWidgets.QCheckBox('Exclude columns where flag 3 is set to True')
        self.chb_flag_4 = QtWidgets.QCheckBox('Exclude columns where flag 4 is set to True')

        # Frame 4
        self.chb_recalculate_graphs = QtWidgets.QCheckBox('Recalculate graph data before compiling data (might be very slow)')

        self.step = 0
        self.state_list = []

        self.set_layout()
        self.exec_()

    def frame_0_layout(self):
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_file)
        v_layout.addWidget(self.rbtn_current_project)
        v_layout.addWidget(self.rbtn_list_of_projects)
        v_layout.addWidget(self.lst_files)
        v_layout.addWidget(self.btn_add_files)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.rbtn_current_project.setChecked(True)
        self.rbtn_list_of_projects.setChecked(False)
        self.lst_files.setDisabled(True)
        self.btn_add_files.setDisabled(True)

        self.lst_files.setMinimumWidth(500)

        self.rbtn_current_project.toggled.connect(self.rbtn_current_project_trigger)
        self.btn_add_files.clicked.connect(self.btn_add_files_trigger)

        self.widget_frame_0.setLayout(h_layout)

    def frame_1_layout(self):

        self.cmb_nominal_data.addItems([
            'advanced_species',
            'atomic_species',
            'zeta',
            'n',
            'flag_1',
            'flag_2',
            'flag_3',
            'flag_4'
        ])

        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_nominal_data)
        v_layout.addWidget(self.cmb_nominal_data)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.widget_frame_1.setLayout(h_layout)

    def frame_2_layout(self):
        self.list_2.addItems([
            'peak_gamma',
            'avg_gamma',
            'theta_min',
            'theta_max',
            'theta_angle_variance',
            'redshift',
            'avg_redshift',
            'avg_central_separation'
        ])

        self.list_1.addItems([
            'normalized_peak_gamma',
            'normalized_avg_gamma',
            'alpha_min',
            'alpha_max',
            'theta_angle_mean'
        ])

        h_layout = QtWidgets.QHBoxLayout()
        v_layout_1 = QtWidgets.QVBoxLayout()
        v_layout_2 = QtWidgets.QVBoxLayout()
        v_layout_3 = QtWidgets.QVBoxLayout()
        v_layout_4 = QtWidgets.QVBoxLayout()
        v_layout_5 = QtWidgets.QVBoxLayout()

        v_layout_1.addStretch()
        v_layout_1.addWidget(self.btn_list_1_up)
        v_layout_1.addWidget(self.btn_list_1_down)
        v_layout_1.addStretch()

        v_layout_2.addWidget(self.lbl_included_data)
        v_layout_2.addWidget(self.list_1)

        v_layout_3.addStretch()
        v_layout_3.addWidget(self.btn_add)
        v_layout_3.addWidget(self.btn_remove)
        v_layout_3.addStretch()

        v_layout_4.addWidget(self.lbl_available_data)
        v_layout_4.addWidget(self.list_2)

        v_layout_5.addStretch()
        v_layout_5.addWidget(self.btn_list_2_up)
        v_layout_5.addWidget(self.btn_list_2_down)
        v_layout_5.addStretch()

        h_layout.addLayout(v_layout_1)
        h_layout.addLayout(v_layout_2)
        h_layout.addLayout(v_layout_3)
        h_layout.addLayout(v_layout_4)
        h_layout.addLayout(v_layout_5)

        self.widget_frame_2.setLayout(h_layout)

    def frame_3_layout(self):
        self.chb_edge_columns.setChecked(True)
        self.chb_matrix_columns.setChecked(False)
        self.chb_particle_columns.setChecked(False)
        self.chb_hidden_columns.setChecked(False)
        self.chb_flag_1.setChecked(False)
        self.chb_flag_2.setChecked(False)
        self.chb_flag_3.setChecked(False)
        self.chb_flag_4.setChecked(False)

        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()

        v_layout.addStretch()
        v_layout.addWidget(self.lbl_filter)
        v_layout.addWidget(self.chb_edge_columns)
        v_layout.addWidget(self.chb_matrix_columns)
        v_layout.addWidget(self.chb_particle_columns)
        v_layout.addWidget(self.chb_hidden_columns)
        v_layout.addWidget(self.chb_flag_1)
        v_layout.addWidget(self.chb_flag_2)
        v_layout.addWidget(self.chb_flag_3)
        v_layout.addWidget(self.chb_flag_4)
        v_layout.addStretch()

        h_layout.addStretch()
        h_layout.addLayout(v_layout)
        h_layout.addStretch()

        self.widget_frame_3.setLayout(h_layout)

    def frame_4_layout(self):
        self.chb_recalculate_graphs.setChecked(False)
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self.chb_recalculate_graphs)
        v_layout.addStretch()
        self.widget_frame_4.setLayout(v_layout)

    def set_layout(self):
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_cancel)
        self.btn_layout.addWidget(self.btn_back)
        self.btn_layout.addWidget(self.btn_next)
        self.btn_layout.addStretch()

        self.frame_0_layout()
        self.stack_layout.addWidget(self.widget_frame_0)

        self.frame_1_layout()
        self.stack_layout.addWidget(self.widget_frame_1)

        self.frame_2_layout()
        self.stack_layout.addWidget(self.widget_frame_2)

        self.frame_3_layout()
        self.stack_layout.addWidget(self.widget_frame_3)

        self.frame_4_layout()
        self.stack_layout.addWidget(self.widget_frame_4)

        self.top_layout.addLayout(self.stack_layout)
        self.top_layout.addLayout(self.btn_layout)

        self.setLayout(self.top_layout)

    def btn_add_files_trigger(self):
        prompt = QtWidgets.QFileDialog()
        prompt.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if prompt.exec_():
            filenames = prompt.selectedFiles()
        else:
            filenames = None
        if filenames is not None:
            for file_ in filenames:
                exlusive = True
                for i in range(self.lst_files.count()):
                    if file_ == self.lst_files.item(i).text():
                        exlusive = False
                if exlusive:
                    self.lst_files.addItem(file_)

    def rbtn_current_project_trigger(self, state):
        self.lst_files.setDisabled(state)
        self.btn_add_files.setDisabled(state)

    def btn_list_2_up_trigger(self):
        if self.list_2.currentItem() is not None:
            text = self.list_2.currentItem().text()
            index = self.list_2.currentRow()
            if not index == 0:
                self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))
                self.list_2.insertItem(index - 1, text)
                self.list_2.setCurrentRow(index - 1)

    def btn_list_2_down_trigger(self):
        if self.list_2.currentItem() is not None:
            text = self.list_2.currentItem().text()
            index = self.list_2.currentRow()
            if not index == self.list_2.count() - 1:
                self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))
                self.list_2.insertItem(index + 1, text)
                self.list_2.setCurrentRow(index + 1)

    def btn_list_1_up_trigger(self):
        if self.list_1.currentItem() is not None:
            text = self.list_1.currentItem().text()
            index = self.list_1.currentRow()
            if not index == 0:
                self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))
                self.list_1.insertItem(index - 1, text)
                self.list_1.setCurrentRow(index - 1)

    def btn_list_1_down_trigger(self):
        if self.list_1.currentItem() is not None:
            text = self.list_1.currentItem().text()
            index = self.list_1.currentRow()
            if not index == self.list_1.count() - 1:
                self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))
                self.list_1.insertItem(index + 1, text)
                self.list_1.setCurrentRow(index + 1)

    def btn_add_item_trigger(self):
        if self.list_2.currentItem() is not None:
            self.list_1.addItem(self.list_2.currentItem().text())
            self.list_2.takeItem(self.list_2.row(self.list_2.currentItem()))

    def btn_remove_item_trigger(self):
        if self.list_1.currentItem() is not None:
            self.list_2.addItem(self.list_1.currentItem().text())
            self.list_1.takeItem(self.list_1.row(self.list_1.currentItem()))

    def chb_custom_categories_trigger(self, state):
        if state:
            self.cmb_categorization.setDisabled(True)
            self.cmb_nominal_data.setDisabled(False)
        else:
            self.cmb_categorization.setDisabled(False)
            self.cmb_nominal_data.setDisabled(True)

    def calc(self):
        self.close()
        self.ui_obj.sys_message('Working...')

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save model as", '', "")
        if filename[0]:
            # Prepare string with filenames
            if self.rbtn_list_of_projects.isChecked():
                files = ''
                for i in range(self.lst_files.count()):
                    if not i == self.lst_files.count() - 1:
                        files += self.lst_files.item(i).text() + '\n'
                    else:
                        files += self.lst_files.item(i).text()
            else:
                files = self.ui_obj.savefile
            # Prepare filter
            filter_ = {
                'exclude_edge_columns': self.chb_edge_columns.isChecked(),
                'exclude_matrix_columns': self.chb_matrix_columns.isChecked(),
                'exclude_particle_columns': self.chb_particle_columns.isChecked(),
                'exclude_hidden_columns': self.chb_hidden_columns.isChecked(),
                'exclude_flag_1_columns': self.chb_flag_1.isChecked(),
                'exclude_flag_2_columns': self.chb_flag_2.isChecked(),
                'exclude_flag_3_columns': self.chb_flag_3.isChecked(),
                'exclude_flag_4_columns': self.chb_flag_4.isChecked()
            }
            # Prepare keys
            attr_keys = []
            for i in range(self.list_1.count()):
                attr_keys.append(self.list_1.item(i).text())
            cat_key = self.cmb_nominal_data.currentText()

            # Initiate manager
            manager = data_module.VertexDataManager(
                files,
                filter_=filter_,
                save_filename=filename[0],
                recalc=self.chb_recalculate_graphs.isChecked(),
                attr_keys=attr_keys,
                category_key=cat_key
            )
            manager.process_data()
            manager.save()
            GUI.logger.info('Successfully saved model parameters to {}'.format(filename[0]))
        self.ui_obj.sys_message('Ready.')

    def get_next_frame(self):
        next_frame = 0
        if self.stack_layout.currentIndex() == 0:
            next_frame = 1
        elif self.stack_layout.currentIndex() == 1:
            next_frame = 2
        elif self.stack_layout.currentIndex() == 2:
            next_frame = 3
        elif self.stack_layout.currentIndex() == 3:
            next_frame = 4
            self.btn_next.setText('Calcuate!')
        elif self.stack_layout.currentIndex() == 4:
            next_frame = -2
        else:
            logger.error('Error!')
            self.close()

        return next_frame

    def get_previous_frame(self):
        previous_frame = 0
        if self.stack_layout.currentIndex() == 0:
            previous_frame = -1
        elif self.stack_layout.currentIndex() == 1:
            previous_frame = 0
        elif self.stack_layout.currentIndex() == 2:
            previous_frame = 1
        elif self.stack_layout.currentIndex() == 3:
            previous_frame = 2
        elif self.stack_layout.currentIndex() == 4:
            previous_frame = 3
        else:
            logger.error('Error!')
            self.close()

        return previous_frame

    def btn_next_trigger(self):
        next_frame = self.get_next_frame()
        if next_frame == -2:
            self.calc()
        elif next_frame == -1:
            self.close()
        else:
            self.stack_layout.setCurrentIndex(next_frame)

    def btn_back_trigger(self):
        previous_frame = self.get_previous_frame()
        if previous_frame == -1:
            self.close()
        else:
            self.stack_layout.setCurrentIndex(previous_frame)

    def btn_cancel_trigger(self):
        self.close()


class PlotModels(QtWidgets.QDialog):

    def __init__(self, *args, ui_obj=None, model=None):
        super().__init__(*args)

        self.ui_obj = ui_obj

        self.setWindowTitle('Display model details')

        if model:
            self.model = data_module.VertexDataManager.load(model)
        else:
            self.model = data_module.VertexDataManager.load('default_model')

        self.lbl_viewing_model = QtWidgets.QLabel('')
        self.lbl_categories = QtWidgets.QLabel('')
        self.lbl_attributes = QtWidgets.QLabel('')
        self.lbl_size = QtWidgets.QLabel('')
        self.lbl_attribute_1 = QtWidgets.QLabel('Attribute 1: ')
        self.lbl_attribute_2 = QtWidgets.QLabel('Attribute 2: ')
        self.lbl_pca_categories = QtWidgets.QLabel('Show categorization: ')
        self.lbl_single_attribute = QtWidgets.QLabel('Attribute: ')

        self.btn_quit = QtWidgets.QPushButton('Quit')
        self.btn_quit.clicked.connect(self.btn_quit_trigger)
        self.btn_activate_model = QtWidgets.QPushButton('Set as active')
        self.btn_activate_model.clicked.connect(self.btn_activate_model_trigger)
        self.btn_activate_model.setDisabled(True)
        self.btn_load_model = QtWidgets.QPushButton('Load model')
        self.btn_load_model.clicked.connect(self.btn_load_model_trigger)
        self.btn_dual_plot = QtWidgets.QPushButton('Plot')
        self.btn_dual_plot.clicked.connect(self.btn_dual_plot_trigger)
        self.btn_plot_all = QtWidgets.QPushButton('Plot fitted normal distributions')
        self.btn_plot_all.clicked.connect(self.btn_plot_all_trigger)
        self.btn_plot_z_scores = QtWidgets.QPushButton('Plot z-scores')
        self.btn_plot_z_scores.clicked.connect(self.btn_plot_z_scores_trigger)
        self.btn_plot_pca = QtWidgets.QPushButton('Plot 2 first PC')
        self.btn_plot_pca.clicked.connect(self.btn_plot_pca_trigger)
        self.btn_plot_all_pca = QtWidgets.QPushButton('Plot all PC distributions')
        self.btn_plot_all_pca.clicked.connect(self.btn_plot_all_pca_trigger)
        self.btn_plot_single = QtWidgets.QPushButton('Plot')
        self.btn_plot_single.clicked.connect(self.btn_plot_single_trigger)
        self.btn_print_details = QtWidgets.QPushButton('Print details')
        self.btn_print_details.clicked.connect(self.btn_print_details_trigger)
        self.btn_custom = QtWidgets.QPushButton('Custom')
        self.btn_custom.clicked.connect(self.btn_custom_trigger)

        self.cmb_attribute_1 = QtWidgets.QComboBox()
        self.cmb_attribute_2 = QtWidgets.QComboBox()
        self.cmb_single_attribute = QtWidgets.QComboBox()
        self.cmb_pca_setting = QtWidgets.QComboBox()

        self.cmb_pca_setting.addItem('Show categories')
        self.cmb_pca_setting.addItem('No categories')

        self.set_layout()
        self.exec_()

    def set_layout(self):
        # Fill info into widgets:
        self.lbl_viewing_model.setText('Model: {}'.format(self.model.save_filename))
        if self.ui_obj.project_instance is not None:
            if self.model.save_filename == self.ui_obj.project_instance.graph.active_model:
                self.btn_activate_model.setDisabled(True)
            else:
                self.btn_activate_model.setDisabled(False)
        else:
            self.btn_activate_model.setDisabled(True)
        number_of_files = 0
        for file in self.model.files.splitlines(keepends=False):
            number_of_files += 1
        gen_string = ''
        gen_string += 'Total size, n = {}\n'.format(self.model.uncategorized_normal_dist.n)
        gen_string += 'Data gathered from {} images\n'.format(number_of_files)
        gen_string += 'Filter:\n'
        for key, value in self.model.filter_.items():
            gen_string += '    {}: {}\n'.format(key, value)
        self.lbl_size.setText(gen_string)
        attr_string = ''
        for attribute in self.model.attribute_keys:
            self.cmb_attribute_1.addItem(attribute)
            self.cmb_attribute_2.addItem(attribute)
            self.cmb_single_attribute.addItem(attribute)
            attr_string += '{}\n'.format(attribute)
        self.lbl_attributes.setText('{}'.format(attr_string))
        cat_string = ''
        for c, category_key in enumerate(self.model.category_list):
            cat_string += '{}\t\tn = {}\n'.format(category_key, self.model.composite_model[c].n)
        self.lbl_categories.setText('{}'.format(cat_string))

        # Create group boxes:
        grp_set_model = QtWidgets.QGroupBox('Select model')
        grp_dual_plot = QtWidgets.QGroupBox('Dual plot')
        grp_plot_all = QtWidgets.QGroupBox('Plot all model attributes densities')
        grp_plot_pca = QtWidgets.QGroupBox('Plot 2 first principle components')
        grp_categorization = QtWidgets.QGroupBox('Data categories ({}):'.format(self.model.num_data_categories))
        grp_attributes = QtWidgets.QGroupBox('Data attributes ({}):'.format(self.model.k))
        grp_general = QtWidgets.QGroupBox('General:')
        grp_single_plot = QtWidgets.QGroupBox('Plot attribute model distributions')

        # Create group layouts:
        grp_set_model_layout = QtWidgets.QVBoxLayout()
        grp_set_model_layout.addWidget(self.lbl_viewing_model)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_activate_model)
        layout.addStretch()
        grp_set_model_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_load_model)
        layout.addStretch()
        grp_set_model_layout.addLayout(layout)
        grp_set_model.setLayout(grp_set_model_layout)

        grp_dual_plot_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lbl_attribute_1)
        layout.addWidget(self.cmb_attribute_1)
        grp_dual_plot_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lbl_attribute_2)
        layout.addWidget(self.cmb_attribute_2)
        grp_dual_plot_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_dual_plot)
        layout.addStretch()
        grp_dual_plot_layout.addLayout(layout)
        grp_dual_plot.setLayout(grp_dual_plot_layout)

        grp_plot_all_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_plot_all)
        layout.addStretch()
        grp_plot_all_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_plot_z_scores)
        layout.addStretch()
        grp_plot_all_layout.addLayout(layout)
        grp_plot_all.setLayout(grp_plot_all_layout)

        grp_plot_pca_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lbl_pca_categories)
        layout.addWidget(self.cmb_pca_setting)
        grp_plot_pca_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_plot_pca)
        layout.addWidget(self.btn_plot_all_pca)
        layout.addStretch()
        grp_plot_pca_layout.addLayout(layout)
        grp_plot_pca.setLayout(grp_plot_pca_layout)

        grp_categorization_layout = QtWidgets.QVBoxLayout()
        grp_categorization_layout.addWidget(self.lbl_categories)
        grp_categorization.setLayout(grp_categorization_layout)

        grp_attributes_layout = QtWidgets.QVBoxLayout()
        grp_attributes_layout.addWidget(self.lbl_attributes)
        grp_attributes.setLayout(grp_attributes_layout)

        grp_single_plot_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lbl_single_attribute)
        layout.addWidget(self.cmb_single_attribute)
        grp_single_plot_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_plot_single)
        layout.addWidget(self.btn_custom)
        layout.addStretch()
        grp_single_plot_layout.addLayout(layout)
        grp_single_plot.setLayout(grp_single_plot_layout)

        grp_general_layout = QtWidgets.QVBoxLayout()
        grp_general_layout.addWidget(self.lbl_size)
        grp_general_layout.addWidget(self.btn_print_details)
        grp_general.setLayout(grp_general_layout)

        # Set top layout:
        grid = QtWidgets.QGridLayout()
        grid.addWidget(grp_set_model, 0, 0)
        grid.addWidget(grp_general, 1, 0)
        grid.addWidget(grp_categorization, 2, 0)
        grid.addWidget(grp_attributes, 3, 0)
        grid.addWidget(grp_single_plot, 0, 1)
        grid.addWidget(grp_dual_plot, 1, 1)
        grid.addWidget(grp_plot_all, 2, 1)
        grid.addWidget(grp_plot_pca, 3, 1)

        top_layout = QtWidgets.QVBoxLayout()
        top_layout.addLayout(grid)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.btn_quit)
        layout.addStretch()
        top_layout.addLayout(layout)

        self.setLayout(top_layout)

    def btn_activate_model_trigger(self):
        self.ui_obj.project_instance.graph.active_model = self.model.save_filename

    def btn_load_model_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Load model", '', "")
        if filename[0]:
            self.close()
            PlotModels(ui_obj=self.ui_obj, model=filename[0])

    def btn_dual_plot_trigger(self):
        self.model.dual_plot(self.cmb_attribute_1.currentText(), self.cmb_attribute_2.currentText())

    def btn_plot_all_trigger(self):
        self.model.plot_all()

    def btn_plot_z_scores_trigger(self):
        self.model.z_plot()

    def btn_plot_pca_trigger(self):
        if self.cmb_pca_setting.currentText() == 'Show categories':
            self.model.plot_pca(show_category=True)
        else:
            self.model.plot_pca(show_category=False)

    def btn_plot_all_pca_trigger(self):
        self.model.plot_all_pc()

    def btn_plot_single_trigger(self):
        self.model.single_plot(self.cmb_single_attribute.currentText())

    def btn_custom_trigger(self):
        self.model.thesis_plot()

    def btn_print_details_trigger(self):
        logger.info(self.model.report())

    def btn_quit_trigger(self):
        self.close()


class CustomizeOverlay(QtWidgets.QDialog):

    def __init__(self, *args, ui_obj=None):
        super().__init__(*args)

        self.ui_obj = ui_obj

        self.setWindowTitle('Customize overlay')

        self.parser = configparser.ConfigParser()
        self.parser.read('config.ini')

        self.example_vertices = [

        ]

        # Dialog buttons:
        self.btn_set_default = QtWidgets.QPushButton('Restore default')
        self.btn_set_default.clicked.connect(self.btn_set_default_trigger)
        self.btn_apply_changes = QtWidgets.QPushButton('Apply changes')
        self.btn_apply_changes.clicked.connect(self.btn_apply_changes_trigger)
        self.btn_cancel_changes = QtWidgets.QPushButton('Cancel')
        self.btn_cancel_changes.clicked.connect(self.btn_cancel_changes_trigger)
        self.btn_overlay_scheme = QtWidgets.QPushButton('Set overlay scheme')
        self.btn_overlay_scheme.clicked.connect(self.btn_overlay_scheme_trigger)

        self.dialog_btn_layout = QtWidgets.QHBoxLayout()
        self.dialog_btn_layout.addStretch()
        self.dialog_btn_layout.addWidget(self.btn_apply_changes)
        self.dialog_btn_layout.addWidget(self.btn_cancel_changes)
        self.dialog_btn_layout.addWidget(self.btn_set_default)
        self.dialog_btn_layout.addWidget(self.btn_overlay_scheme)
        self.dialog_btn_layout.addStretch()

        # Layout elements:
        self.cmb_categorization = QtWidgets.QComboBox()
        self.cmb_categorization.addItem('Simple')

        # Simple categorization:
        self.lbl_species = QtWidgets.QLabel('Species:')
        self.lbl_si = QtWidgets.QLabel('Si')
        self.lbl_cu = QtWidgets.QLabel('Cu')
        self.lbl_al = QtWidgets.QLabel('Al')
        self.lbl_mg = QtWidgets.QLabel('Mg')
        self.lbl_un = QtWidgets.QLabel('Un')

        self.lbl_shape = QtWidgets.QLabel('Shape:')
        self.lbl_color_1 = QtWidgets.QLabel('Primary color:')
        self.lbl_color_2 = QtWidgets.QLabel('Secondary color:')
        self.lbl_radius_1 = QtWidgets.QLabel('Primary radius (nm):')
        self.lbl_radius_2 = QtWidgets.QLabel('Secondary radius (%):')
        self.lbl_visible = QtWidgets.QLabel('Show:')
        self.btn_preview = QtWidgets.QPushButton('Refresh preview')
        self.btn_preview.clicked.connect(self.btn_preview_trigger)
        self.lbl_zeta_0 = QtWidgets.QLabel('zeta=0')
        self.lbl_zeta_1 = QtWidgets.QLabel('zeta=1')

        self.cmb_si_shape = QtWidgets.QComboBox()
        self.cmb_si_shape.addItem('Circle')
        self.cmb_cu_shape = QtWidgets.QComboBox()
        self.cmb_cu_shape.addItem('Circle')
        self.cmb_al_shape = QtWidgets.QComboBox()
        self.cmb_al_shape.addItem('Circle')
        self.cmb_mg_shape = QtWidgets.QComboBox()
        self.cmb_mg_shape.addItem('Circle')
        self.cmb_un_shape = QtWidgets.QComboBox()
        self.cmb_un_shape.addItem('Circle')

        self.si_color_1_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'si_primary_r'),
            g=self.parser.getint('colors', 'si_primary_g'),
            b=self.parser.getint('colors', 'si_primary_b'),
            a=self.parser.getint('colors', 'si_primary_a')
        )
        self.cu_color_1_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'cu_primary_r'),
            g=self.parser.getint('colors', 'cu_primary_g'),
            b=self.parser.getint('colors', 'cu_primary_b'),
            a=self.parser.getint('colors', 'cu_primary_a')
        )
        self.al_color_1_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'al_primary_r'),
            g=self.parser.getint('colors', 'al_primary_g'),
            b=self.parser.getint('colors', 'al_primary_b'),
            a=self.parser.getint('colors', 'al_primary_a')
        )
        self.mg_color_1_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'mg_primary_r'),
            g=self.parser.getint('colors', 'mg_primary_g'),
            b=self.parser.getint('colors', 'mg_primary_b'),
            a=self.parser.getint('colors', 'mg_primary_a')
        )
        self.un_color_1_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'un_primary_r'),
            g=self.parser.getint('colors', 'un_primary_g'),
            b=self.parser.getint('colors', 'un_primary_b'),
            a=self.parser.getint('colors', 'un_primary_a')
        )

        self.si_color_2_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'si_secondary_r'),
            g=self.parser.getint('colors', 'si_secondary_g'),
            b=self.parser.getint('colors', 'si_secondary_b'),
            a=self.parser.getint('colors', 'si_secondary_a')
        )
        self.cu_color_2_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'cu_secondary_r'),
            g=self.parser.getint('colors', 'cu_secondary_g'),
            b=self.parser.getint('colors', 'cu_secondary_b'),
            a=self.parser.getint('colors', 'cu_secondary_a')
        )
        self.al_color_2_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'al_secondary_r'),
            g=self.parser.getint('colors', 'al_secondary_g'),
            b=self.parser.getint('colors', 'al_secondary_b'),
            a=self.parser.getint('colors', 'al_secondary_a')
        )
        self.mg_color_2_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'mg_secondary_r'),
            g=self.parser.getint('colors', 'mg_secondary_g'),
            b=self.parser.getint('colors', 'mg_secondary_b'),
            a=self.parser.getint('colors', 'mg_secondary_a')
        )
        self.un_color_2_selector = GUI_custom_components.RgbaSelector(
            r=self.parser.getint('colors', 'un_secondary_r'),
            g=self.parser.getint('colors', 'un_secondary_g'),
            b=self.parser.getint('colors', 'un_secondary_b'),
            a=self.parser.getint('colors', 'un_secondary_a')
        )

        self.si_size_1 = QtWidgets.QDoubleSpinBox()
        self.si_size_1.setMaximum(400.00)
        self.si_size_1.setMinimum(0.00)
        self.si_size_1.setDecimals(2)
        self.si_size_1.setValue(self.parser.getfloat('overlay_radii', 'si_outer'))
        self.si_size_2 = QtWidgets.QSpinBox()
        self.si_size_2.setMaximum(60)
        self.si_size_2.setMinimum(0)
        self.si_size_2.setValue(self.parser.getint('overlay_radii', 'si_inner'))

        self.cu_size_1 = QtWidgets.QDoubleSpinBox()
        self.cu_size_1.setMaximum(400.00)
        self.cu_size_1.setMinimum(0.00)
        self.cu_size_1.setDecimals(2)
        self.cu_size_1.setValue(self.parser.getfloat('overlay_radii', 'cu_outer'))
        self.cu_size_2 = QtWidgets.QSpinBox()
        self.cu_size_2.setMaximum(60)
        self.cu_size_2.setMinimum(0)
        self.cu_size_2.setValue(self.parser.getint('overlay_radii', 'cu_inner'))

        self.al_size_1 = QtWidgets.QDoubleSpinBox()
        self.al_size_1.setMaximum(400.00)
        self.al_size_1.setMinimum(0.00)
        self.al_size_1.setDecimals(2)
        self.al_size_1.setValue(self.parser.getfloat('overlay_radii', 'al_outer'))
        self.al_size_2 = QtWidgets.QSpinBox()
        self.al_size_2.setMaximum(60)
        self.al_size_2.setMinimum(0)
        self.al_size_2.setValue(self.parser.getint('overlay_radii', 'al_inner'))

        self.mg_size_1 = QtWidgets.QDoubleSpinBox()
        self.mg_size_1.setMaximum(400.00)
        self.mg_size_1.setMinimum(0.00)
        self.mg_size_1.setDecimals(2)
        self.mg_size_1.setValue(self.parser.getfloat('overlay_radii', 'mg_outer'))
        self.mg_size_2 = QtWidgets.QSpinBox()
        self.mg_size_2.setMaximum(60)
        self.mg_size_2.setMinimum(0)
        self.mg_size_2.setValue(self.parser.getint('overlay_radii', 'mg_inner'))

        self.un_size_1 = QtWidgets.QDoubleSpinBox()
        self.un_size_1.setMaximum(400.00)
        self.un_size_1.setMinimum(0.00)
        self.un_size_1.setDecimals(2)
        self.un_size_1.setValue(self.parser.getfloat('overlay_radii', 'un_outer'))
        self.un_size_2 = QtWidgets.QSpinBox()
        self.un_size_2.setMaximum(60)
        self.un_size_2.setMinimum(0)
        self.un_size_2.setValue(self.parser.getint('overlay_radii', 'un_inner'))

        self.chb_si_show = QtWidgets.QCheckBox('')
        self.chb_si_show.setChecked(True)
        self.chb_cu_show = QtWidgets.QCheckBox('')
        self.chb_cu_show.setChecked(True)
        self.chb_al_show = QtWidgets.QCheckBox('')
        self.chb_al_show.setChecked(True)
        self.chb_mg_show = QtWidgets.QCheckBox('')
        self.chb_mg_show.setChecked(True)
        self.chb_un_show = QtWidgets.QCheckBox('')
        self.chb_un_show.setChecked(True)

        self.gv_list = []

        categories = ['si', 'cu', 'al', 'mg', 'un']

        for s, species in enumerate(categories):
            self.gv_list.append([])
            self.gv_list[s].append(QtWidgets.QGraphicsView(QtWidgets.QGraphicsScene()))
            self.gv_list[s].append(QtWidgets.QGraphicsView(QtWidgets.QGraphicsScene()))

        self.redraw_vertices()

        for s, species in enumerate(categories):
            self.gv_list[s][0].setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.gv_list[s][0].setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.gv_list[s][0].scale(2.5, 2.5)
            self.gv_list[s][1].setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.gv_list[s][1].setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.gv_list[s][1].scale(2.5, 2.5)

        self.set_layout()
        self.exec_()

    def redraw_vertices(self):

        zeta = [0, 1]
        categories = ['si', 'cu', 'al', 'mg', 'un']

        for s, species in enumerate(categories):
            color_1 = QtGui.QColor(
                eval('self.{}_color_1_selector.r_box.value()'.format(species)),
                eval('self.{}_color_1_selector.g_box.value()'.format(species)),
                eval('self.{}_color_1_selector.b_box.value()'.format(species)),
            )
            for z in zeta:
                if z == 0:
                    color_2 = color_1
                else:
                    color_2 = QtGui.QColor(
                        eval('self.{}_color_2_selector.r_box.value()'.format(species)),
                        eval('self.{}_color_2_selector.g_box.value()'.format(species)),
                        eval('self.{}_color_2_selector.b_box.value()'.format(species)),
                    )
                pen = QtGui.QPen(color_1)
                r = eval('self.{}_size_1.value()'.format(species)) / 10
                r_factor = eval('self.{}_size_2.value()'.format(species)) / 10
                pen.setWidth(r_factor)
                brush = QtGui.QBrush(color_2)

                vertex_graphic = GUI_custom_components.InteractiveOverlayColumn(
                    ui_obj=self.ui_obj,
                    vertex=None,
                    r=r,
                    scale_factor=1,
                    movable=False,
                    selectable=False,
                    pen=pen,
                    brush=brush
                )
                vertex_graphic.i = self.ui_obj.selected_column + 1
                vertex_graphic.set_style()
                for item in self.gv_list[s][z].scene().items():
                    self.gv_list[s][z].scene().removeItem(item)

                self.gv_list[s][z].scene().addItem(vertex_graphic)

    def set_layout(self):
        grid_layout = QtWidgets.QGridLayout()

        strut_1 = QtWidgets.QWidget()
        strut_1.setMinimumWidth(100)
        strut_1.setMaximumHeight(0)
        strut_2 = QtWidgets.QWidget()
        strut_2.setMinimumWidth(150)
        strut_2.setMaximumHeight(0)
        strut_3 = QtWidgets.QWidget()
        strut_3.setMinimumWidth(20)
        strut_3.setMaximumHeight(0)
        strut_4 = QtWidgets.QWidget()
        strut_4.setMinimumWidth(20)
        strut_4.setMaximumHeight(0)
        strut_5 = QtWidgets.QWidget()
        strut_5.setMinimumWidth(100)
        strut_5.setMaximumHeight(0)
        strut_6 = QtWidgets.QWidget()
        strut_6.setMinimumWidth(100)
        strut_6.setMaximumHeight(0)
        strut_7 = QtWidgets.QWidget()
        strut_7.setMinimumWidth(80)
        strut_7.setMaximumHeight(0)
        strut_8 = QtWidgets.QWidget()
        strut_8.setMinimumWidth(80)
        strut_8.setMaximumHeight(0)

        v_strut = QtWidgets.QWidget()
        strut_8.setMaximumWidth(0)
        strut_8.setMinimumHeight(50)

        # Column 1:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_species), 0, 0)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_si), 1, 0)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_cu), 2, 0)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_al), 3, 0)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_mg), 4, 0)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_un), 5, 0)
        grid_layout.addWidget(strut_1, 6, 0)
        grid_layout.addWidget(v_strut, 7, 0)

        # Column 2:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_shape), 0, 1)
        grid_layout.addWidget(self.cmb_si_shape, 1, 1)
        grid_layout.addWidget(self.cmb_cu_shape, 2, 1)
        grid_layout.addWidget(self.cmb_al_shape, 3, 1)
        grid_layout.addWidget(self.cmb_mg_shape, 4, 1)
        grid_layout.addWidget(self.cmb_un_shape, 5, 1)
        grid_layout.addWidget(strut_2, 6, 1)

        # Column 3:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_color_1), 0, 2)
        grid_layout.addWidget(self.si_color_1_selector, 1, 2)
        grid_layout.addWidget(self.cu_color_1_selector, 2, 2)
        grid_layout.addWidget(self.al_color_1_selector, 3, 2)
        grid_layout.addWidget(self.mg_color_1_selector, 4, 2)
        grid_layout.addWidget(self.un_color_1_selector, 5, 2)
        grid_layout.addWidget(strut_3, 6, 2)

        # Column 4:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_color_2), 0, 3)
        grid_layout.addWidget(self.si_color_2_selector, 1, 3)
        grid_layout.addWidget(self.cu_color_2_selector, 2, 3)
        grid_layout.addWidget(self.al_color_2_selector, 3, 3)
        grid_layout.addWidget(self.mg_color_2_selector, 4, 3)
        grid_layout.addWidget(self.un_color_2_selector, 5, 3)
        grid_layout.addWidget(strut_4, 6, 3)

        # Column 5:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_radius_1), 0, 4)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.si_size_1), 1, 4)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.cu_size_1), 2, 4)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.al_size_1), 3, 4)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.mg_size_1), 4, 4)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.un_size_1), 5, 4)
        grid_layout.addWidget(strut_5, 6, 4)

        # Column 6:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_radius_2), 0, 5)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.si_size_2), 1, 5)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.cu_size_2), 2, 5)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.al_size_2), 3, 5)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.mg_size_2), 4, 5)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.un_size_2), 5, 5)
        grid_layout.addWidget(strut_6, 6, 5)

        # Column 7:
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_visible), 0, 6)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.chb_si_show), 1, 6)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.chb_cu_show), 2, 6)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.chb_al_show), 3, 6)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.chb_mg_show), 4, 6)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.chb_un_show), 5, 6)
        grid_layout.addWidget(strut_7, 6, 6)

        # Column 8/9:
        grid_layout.addWidget(self.btn_preview, 0, 7, 1, 8)

        grid_layout.addWidget(self.gv_list[0][0], 1, 7)
        grid_layout.addWidget(self.gv_list[0][1], 1, 8)
        grid_layout.addWidget(self.gv_list[1][0], 2, 7)
        grid_layout.addWidget(self.gv_list[1][1], 2, 8)
        grid_layout.addWidget(self.gv_list[2][0], 3, 7)
        grid_layout.addWidget(self.gv_list[2][1], 3, 8)
        grid_layout.addWidget(self.gv_list[3][0], 4, 7)
        grid_layout.addWidget(self.gv_list[3][1], 4, 8)
        grid_layout.addWidget(self.gv_list[4][0], 5, 7)
        grid_layout.addWidget(self.gv_list[4][1], 5, 8)

        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_zeta_0), 6, 7)
        grid_layout.addWidget(GUI_custom_components.CenterBox(widget=self.lbl_zeta_1), 6, 8)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(grid_layout)
        layout.addLayout(self.dialog_btn_layout)

        self.setLayout(layout)

    def btn_preview_trigger(self):
        self.redraw_vertices()

    def btn_set_default_trigger(self):
        pass

    def btn_overlay_scheme_trigger(self):
        pass

    def btn_apply_changes_trigger(self):

        self.parser.set('shapes', 'si_shape', str(self.cmb_si_shape.currentText()))
        self.parser.set('shapes', 'cu_shape', str(self.cmb_cu_shape.currentText()))
        self.parser.set('shapes', 'al_shape', str(self.cmb_al_shape.currentText()))
        self.parser.set('shapes', 'mg_shape', str(self.cmb_mg_shape.currentText()))
        self.parser.set('shapes', 'un_shape', str(self.cmb_un_shape.currentText()))

        self.parser.set('colors', 'si_primary_r', str(self.si_color_1_selector.r_box.value()))
        self.parser.set('colors', 'si_primary_g', str(self.si_color_1_selector.g_box.value()))
        self.parser.set('colors', 'si_primary_b', str(self.si_color_1_selector.b_box.value()))
        self.parser.set('colors', 'si_primary_a', str(self.si_color_1_selector.a_box.value()))
        self.parser.set('colors', 'si_secondary_r', str(self.si_color_2_selector.r_box.value()))
        self.parser.set('colors', 'si_secondary_g', str(self.si_color_2_selector.g_box.value()))
        self.parser.set('colors', 'si_secondary_b', str(self.si_color_2_selector.b_box.value()))
        self.parser.set('colors', 'si_secondary_a', str(self.si_color_2_selector.a_box.value()))

        self.parser.set('colors', 'cu_primary_r', str(self.cu_color_1_selector.r_box.value()))
        self.parser.set('colors', 'cu_primary_g', str(self.cu_color_1_selector.g_box.value()))
        self.parser.set('colors', 'cu_primary_b', str(self.cu_color_1_selector.b_box.value()))
        self.parser.set('colors', 'cu_primary_a', str(self.cu_color_1_selector.a_box.value()))
        self.parser.set('colors', 'cu_secondary_r', str(self.cu_color_2_selector.r_box.value()))
        self.parser.set('colors', 'cu_secondary_g', str(self.cu_color_2_selector.g_box.value()))
        self.parser.set('colors', 'cu_secondary_b', str(self.cu_color_2_selector.b_box.value()))
        self.parser.set('colors', 'cu_secondary_a', str(self.cu_color_2_selector.a_box.value()))

        self.parser.set('colors', 'al_primary_r', str(self.al_color_1_selector.r_box.value()))
        self.parser.set('colors', 'al_primary_g', str(self.al_color_1_selector.g_box.value()))
        self.parser.set('colors', 'al_primary_b', str(self.al_color_1_selector.b_box.value()))
        self.parser.set('colors', 'al_primary_a', str(self.al_color_1_selector.a_box.value()))
        self.parser.set('colors', 'al_secondary_r', str(self.al_color_2_selector.r_box.value()))
        self.parser.set('colors', 'al_secondary_g', str(self.al_color_2_selector.g_box.value()))
        self.parser.set('colors', 'al_secondary_b', str(self.al_color_2_selector.b_box.value()))
        self.parser.set('colors', 'al_secondary_a', str(self.al_color_2_selector.a_box.value()))

        self.parser.set('colors', 'mg_primary_r', str(self.mg_color_1_selector.r_box.value()))
        self.parser.set('colors', 'mg_primary_g', str(self.mg_color_1_selector.g_box.value()))
        self.parser.set('colors', 'mg_primary_b', str(self.mg_color_1_selector.b_box.value()))
        self.parser.set('colors', 'mg_primary_a', str(self.mg_color_1_selector.a_box.value()))
        self.parser.set('colors', 'mg_secondary_r', str(self.mg_color_2_selector.r_box.value()))
        self.parser.set('colors', 'mg_secondary_g', str(self.mg_color_2_selector.g_box.value()))
        self.parser.set('colors', 'mg_secondary_b', str(self.mg_color_2_selector.b_box.value()))
        self.parser.set('colors', 'mg_secondary_a', str(self.mg_color_2_selector.a_box.value()))

        self.parser.set('colors', 'un_primary_r', str(self.un_color_1_selector.r_box.value()))
        self.parser.set('colors', 'un_primary_g', str(self.un_color_1_selector.g_box.value()))
        self.parser.set('colors', 'un_primary_b', str(self.un_color_1_selector.b_box.value()))
        self.parser.set('colors', 'un_primary_a', str(self.un_color_1_selector.a_box.value()))
        self.parser.set('colors', 'un_secondary_r', str(self.un_color_2_selector.r_box.value()))
        self.parser.set('colors', 'un_secondary_g', str(self.un_color_2_selector.g_box.value()))
        self.parser.set('colors', 'un_secondary_b', str(self.un_color_2_selector.b_box.value()))
        self.parser.set('colors', 'un_secondary_a', str(self.un_color_2_selector.a_box.value()))

        self.parser.set('overlay_radii', 'si_outer', str(self.si_size_1.value()))
        self.parser.set('overlay_radii', 'cu_outer', str(self.cu_size_1.value()))
        self.parser.set('overlay_radii', 'al_outer', str(self.al_size_1.value()))
        self.parser.set('overlay_radii', 'mg_outer', str(self.mg_size_1.value()))
        self.parser.set('overlay_radii', 'un_outer', str(self.un_size_1.value()))

        self.parser.set('overlay_radii', 'si_inner', str(self.si_size_2.value()))
        self.parser.set('overlay_radii', 'cu_inner', str(self.cu_size_2.value()))
        self.parser.set('overlay_radii', 'al_inner', str(self.al_size_2.value()))
        self.parser.set('overlay_radii', 'mg_inner', str(self.mg_size_2.value()))
        self.parser.set('overlay_radii', 'un_inner', str(self.un_size_2.value()))

        with open('config.ini', 'w') as configfile:
            self.parser.write(configfile)

        self.btn_preview_trigger()

        self.ui_obj.update_overlay()

        logger.info('Updated settings')

    def btn_cancel_changes_trigger(self):
        self.close()


class CustomizeClassification(QtWidgets.QDialog):

    def __init__(self, *args, ui_obj=None, dict_=None):
        super().__init__(*args)

        self.ui_obj = ui_obj
        if self.ui_obj.project_instance is None:
            if dict_ is None:
                self.dict = GUI.core.Project.default_dict
            else:
                self.dict = dict_
        else:
            if dict_ is None:
                self.dict = self.ui_obj.project_instance.species_dict
            else:
                self.dict = dict_

        self.setWindowTitle('Customize classification')

        self.btn_what_is_this = QtWidgets.QPushButton('What is this?')
        self.btn_what_is_this.clicked.connect(self.btn_what_is_this_trigger)

        self.btn_remove = QtWidgets.QPushButton('Remove class')
        self.btn_remove.clicked.connect(self.btn_remove_trigger)
        self.btn_edit = QtWidgets.QPushButton('Edit class')
        self.btn_edit.clicked.connect(self.btn_edit_trigger)
        self.btn_new = QtWidgets.QPushButton('Add new class')
        self.btn_new.clicked.connect(self.btn_new_trigger)

        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.btn_cancel_trigger)
        self.btn_apply = QtWidgets.QPushButton('Apply')
        self.btn_apply.clicked.connect(self.btn_apply_trigger)
        self.btn_set_default = QtWidgets.QPushButton('Set default')
        self.btn_set_default.clicked.connect(self.btn_set_default_trigger)

        self.edit_btn_layout = QtWidgets.QHBoxLayout()
        self.edit_btn_layout.addStretch()
        self.edit_btn_layout.addWidget(self.btn_new)
        self.edit_btn_layout.addWidget(self.btn_edit)
        self.edit_btn_layout.addWidget(self.btn_remove)
        self.edit_btn_layout.addStretch()

        self.dialog_btn_layout = QtWidgets.QHBoxLayout()
        self.dialog_btn_layout.addStretch()
        self.dialog_btn_layout.addWidget(self.btn_apply)
        self.dialog_btn_layout.addWidget(self.btn_cancel)
        self.dialog_btn_layout.addWidget(self.btn_set_default)
        self.dialog_btn_layout.addStretch()

        self.lst_classes = QtWidgets.QListWidget()
        self.lst_classes.itemClicked.connect(self.lst_classes_trigger)
        self.headers = ['Symmetry', 'Atomic species', 'Atomic radii', 'Color (r, g, b)', 'Species color (r, g, b)', 'Description']
        self.fields = ['symmetry', 'atomic_species', 'atomic_radii', 'color', 'species_color', 'description']
        display_string = 'Class: \n'
        for field in self.headers:
            display_string += '{}: \n'.format(field)
        self.lbl_details = QtWidgets.QLabel(display_string)

        self.set_layout()
        self.exec_()

    def update_string(self):
        if self.lst_classes.currentIndex() is None:
            display_string = 'Class: \n'
            for header, field in zip(self.headers, self.fields):
                display_string += '{}: \n'.format(header)
            self.lbl_details.setText(display_string)
        else:
            class_ = self.lst_classes.currentItem().text()
            display_string = 'Class: {}\n'.format(class_)
            for header, field in zip(self.headers, self.fields):
                display_string += '{}: {}\n'.format(header, self.dict[class_][field])
            self.lbl_details.setText(display_string)

    def set_layout(self):
        self.lst_classes = QtWidgets.QListWidget()
        self.lst_classes.itemClicked.connect(self.lst_classes_trigger)
        for class_ in self.dict:
            self.lst_classes.addItem(class_)

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self.btn_what_is_this)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lst_classes)
        layout_2 = QtWidgets.QVBoxLayout()
        layout_2.addLayout(self.edit_btn_layout)
        layout_2.addWidget(self.lbl_details)
        layout.addLayout(layout_2)
        v_layout.addLayout(layout)
        v_layout.addLayout(self.dialog_btn_layout)
        self.setLayout(v_layout)

    def lst_classes_trigger(self):
        self.update_string()

    def btn_apply_trigger(self):
        self.close()
        if self.ui_obj.project_instance is not None:
            self.ui_obj.project_instance.species_dict = self.dict
            self.ui_obj.project_instance.graph.species_dict = self.dict
        else:
            logger.info('Cannot change default categories! Open a project first')

    def btn_cancel_trigger(self):
        self.close()

    def btn_new_trigger(self):
        pass

    def btn_edit_trigger(self):
        CategoryEdit(parent=self)
        self.update_string()

    def btn_set_default_trigger(self):
        self.dict = GUI.core.Project.default_dict
        self.set_layout()

    def btn_remove_trigger(self):
        self.dict.pop(self.lst_classes.currentItem().text())
        self.close()
        CustomizeClassification(ui_obj=self.ui_obj, dict_=self.dict)

    def btn_what_is_this_trigger(self):
        pass


class CategoryEdit(QtWidgets.QDialog):

    def __init__(self, *args, parent=None):
        super().__init__(*args)

        self.parent = parent
        self.class_ = parent.lst_classes.currentItem().text()

        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(self.btn_ok_trigger)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.btn_cancel_trigger)

        self.class_text = QtWidgets.QLineEdit('')
        self.symmetry_box = QtWidgets.QSpinBox()
        self.symmetry_box.setMaximum(10)
        self.symmetry_box.setMinimum(0)
        self.species_text = QtWidgets.QLineEdit('')
        self.radii_box = QtWidgets.QDoubleSpinBox()
        self.radii_box.setMinimum(0.00)
        self.radii_box.setMaximum(500.0)
        self.color_edit = GUI_custom_components.RgbaSelector()
        self.species_color_edit = GUI_custom_components.RgbaSelector()
        self.description_text = QtWidgets.QLineEdit('')

        self.set_layout()
        self.populate_values()
        self.exec_()

    def populate_values(self):
        symmetry = self.parent.dict[self.class_]['symmetry']
        species = self.parent.dict[self.class_]['atomic_species']
        radii = self.parent.dict[self.class_]['atomic_radii']
        color = self.parent.dict[self.class_]['color']
        species_color = self.parent.dict[self.class_]['species_color']
        description = self.parent.dict[self.class_]['description']
        self.class_text.setText(self.class_)
        self.symmetry_box.setValue(symmetry)
        self.species_text.setText(species)
        self.radii_box.setValue(radii)
        self.color_edit.r_box.setValue(color[0])
        self.color_edit.g_box.setValue(color[1])
        self.color_edit.b_box.setValue(color[2])
        self.species_color_edit.r_box.setValue(species_color[0])
        self.species_color_edit.g_box.setValue(species_color[1])
        self.species_color_edit.b_box.setValue(species_color[2])
        self.description_text.setText(description)

    def set_layout(self):
        v_layout = QtWidgets.QVBoxLayout()

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel('Class:'), 0, 0)
        grid.addWidget(QtWidgets.QLabel('Symmetry:'), 0, 1)
        grid.addWidget(QtWidgets.QLabel('Atomic species:'), 0, 2)
        grid.addWidget(QtWidgets.QLabel('Radii:'), 0, 3)
        grid.addWidget(QtWidgets.QLabel('Color:'), 0, 4)
        grid.addWidget(QtWidgets.QLabel('Species color:'), 0, 5)
        grid.addWidget(QtWidgets.QLabel('Description:'), 0, 6)

        grid.addWidget(self.class_text, 1, 0)
        grid.addWidget(self.symmetry_box, 1, 1)
        grid.addWidget(self.species_text, 1, 2)
        grid.addWidget(self.radii_box, 1, 3)
        grid.addWidget(self.color_edit, 1, 4)
        grid.addWidget(self.species_color_edit, 1, 5)
        grid.addWidget(self.description_text, 1, 6)

        v_layout.addLayout(grid)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()

        v_layout.addLayout(btn_layout)

        self.setLayout(v_layout)

    def btn_ok_trigger(self):
        self.close()
        self.parent.dict[self.class_]['symmetry'] = self.symmetry_box.value()
        self.parent.dict[self.class_]['atomic_species'] = self.species_text.text()
        self.parent.dict[self.class_]['atomic_radii'] = self.radii_box.value()
        self.parent.dict[self.class_]['color'] = self.color_edit.get_triple()
        self.parent.dict[self.class_]['species_color'] = self.species_color_edit.get_triple()
        self.parent.dict[self.class_]['description'] = self.description_text.text()

    def btn_cancel_trigger(self):
        self.close()














