import numpy as np
import cv2


class Proyector:

    def __init__(self, P, R0, V2C):
        R0_padded = np.column_stack([np.vstack([R0, [0, 0, 0]]), [0, 0, 0, 1]])
        V2C_padded = np.vstack((V2C, [0, 0, 0, 1]))
        self.trafo_matrix = np.dot(P, np.dot(R0_padded, V2C_padded))

    def project_and_filter_boxes(self, boxes_3d, img):
        h, w, _ = img.shape
        n = boxes_3d.shape[0]
        projected_boxes = []

        for i in range(n):
            corners_3d = boxes_3d[i]  # (8, 3)
            # Añadir coordenada homogénea
            corners_hom = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)
            # Proyección
            pts_2d_homo = (self.trafo_matrix @ corners_hom.T).T  # (8, 3)
            pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]  # Normalizar por Z

            # Verificar que todos los puntos están dentro de la imagen
            inside = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
            )
            if np.all(inside):
                projected_boxes.append(pts_2d)

        return np.array(projected_boxes, dtype=np.float32)  # (m, 8, 2)

    def draw_projected_boxes(self, img, projected_boxes, color=(0, 255, 0), thickness=2):

        img_out = img.copy()

        # Índices que conectan las esquinas para dibujar una bbox 3D
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # base
            (4, 5), (5, 6), (6, 7), (7, 4),  # techo
            (0, 4), (1, 5), (2, 6), (3, 7)   # columnas
        ]

        for box in projected_boxes:
            box_int = box.astype(int)  # Convertir a enteros para OpenCV
            
            # Dibujar las líneas de la caja
            for i1, i2 in edges:
                pt1 = tuple(box_int[i1])
                pt2 = tuple(box_int[i2])
                cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
            
            # Opcional: dibujar las esquinas como círculos
            for (x, y) in box_int:
                cv2.circle(img_out, (x, y), 3, (0, 0, 255), -1)

        return img_out

    def proyect(self, points, image):
        proyected_bboxes = self.project_and_filter_boxes(points, image)
        img_with_bboxes = self.draw_projected_boxes(image, proyected_bboxes)

        return img_with_bboxes
