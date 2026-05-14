import argparse
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree

from object_config import (
    get_cad_models_dir,
    get_default_cad_path,
    get_reconstruction_dir,
)

# ---------------- ЛОГИРОВАНИЕ ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)


class MeshComparator:
    """Сравнение точек реконструкции с CAD: выравнивание масштаба и положения, отклонения точка–поверхность."""

    def __init__(self, cad_path: str, scan_path):
        logging.info("Загрузка моделей...")
        self.cad_mesh = self._load_cad_mesh(cad_path)
        self.scan_pcd, self.scan_mesh = self._load_scan(scan_path)

        logging.info(
            f"CAD: {len(self.cad_mesh.vertices)} вершин, "
            f"{len(self.cad_mesh.triangles)} треугольников"
        )
        n_pts = len(np.asarray(self.scan_pcd.points))
        logging.info(f"Реконструкция (точки): {n_pts}")

    def _load_cad_mesh(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        if path.endswith(".stl") or path.endswith(".obj"):
            mesh = o3d.io.read_triangle_mesh(path)
        else:
            tm = trimesh.load(path)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tm.vertices),
                o3d.utility.Vector3iVector(tm.faces),
            )
        mesh.compute_vertex_normals()
        return mesh

    def _load_scan(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        if path.endswith(".npy"):
            logging.info("Загрузка реконструкции как облако точек (без Poisson)...")
            points = np.load(path)
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError("NPY: ожидается (N,3) или (N,6)")
            points = points[:, :3].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            return pcd, None

        # STL/OBJ — загружаем как меш и как облако вершин для единого пайплайна
        if path.endswith(".stl") or path.endswith(".obj"):
            mesh = o3d.io.read_triangle_mesh(path)
        else:
            tm = trimesh.load(path)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tm.vertices),
                o3d.utility.Vector3iVector(tm.faces),
            )
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        return pcd, mesh

    def preprocess_meshes(self, cad_simplify=None, scale_mode="fit"):
        logging.info("Предобработка...")
        self.cad_mesh.remove_duplicated_vertices()
        self.cad_mesh.remove_duplicated_triangles()
        self.cad_mesh.remove_degenerate_triangles()
        self.cad_mesh.compute_vertex_normals()
        if cad_simplify and len(self.cad_mesh.triangles) > cad_simplify:
            logging.info(f"Упрощение CAD: {len(self.cad_mesh.triangles)} -> {cad_simplify}")
            self.cad_mesh = self.cad_mesh.simplify_quadric_decimation(cad_simplify)
            self.cad_mesh.compute_vertex_normals()
        if self.scan_mesh is not None:
            self.scan_mesh.remove_duplicated_vertices()
            self.scan_mesh.remove_duplicated_triangles()
            self.scan_mesh.remove_degenerate_triangles()
            self.scan_mesh.compute_vertex_normals()
        self._center_and_scale_to_cad(scale_mode=scale_mode)

    def _center_and_scale_to_cad(self, scale_mode="fit"):
        """Центрирование по центру масс, масштаб под CAD, выравнивание основания по вертикали.
        scale_mode: 'fit' — вписать реконструкцию в габариты CAD (углы не выходят за пределы);
                    'median' — по медиане расстояний (как раньше), без дополнительного запаса.
        """
        cad_verts = np.asarray(self.cad_mesh.vertices)
        cad_center = np.mean(cad_verts, axis=0)
        self.cad_mesh.translate(-cad_center)

        scan_pts = np.asarray(self.scan_pcd.points)
        scan_center = np.mean(scan_pts, axis=0)
        self.scan_pcd.translate(-scan_center)
        if self.scan_mesh is not None:
            self.scan_mesh.translate(-scan_center)

        scan_pts = np.asarray(self.scan_pcd.points)
        cad_verts = np.asarray(self.cad_mesh.vertices)
        cad_extent = np.ptp(cad_verts, axis=0)
        scan_extent = np.ptp(scan_pts, axis=0)
        scan_extent = np.maximum(scan_extent, 1e-9)

        if scale_mode == "fit":
            # Масштаб «вписать»: реконструкция не выходит за габариты CAD
            # Без искусственного уменьшения, чтобы точки доходили до краёв модели.
            scale_per_axis = cad_extent / scan_extent
            scale = float(np.min(scale_per_axis))
            logging.info(f"Масштаб реконструкции -> CAD: {scale:.6f} (вписание в габариты без запаса)")
        else:
            cad_r = np.linalg.norm(cad_verts, axis=1)
            scan_r = np.linalg.norm(scan_pts, axis=1)
            cad_median_r = np.median(cad_r)
            scan_median_r = np.median(scan_r)
            scale = float(cad_median_r / scan_median_r) if (scan_median_r > 1e-12 and cad_median_r > 1e-12) else 1.0
            logging.info(f"Масштаб реконструкции -> CAD: {scale:.6f} (медиана расстояний)")

        if abs(scale - 1.0) > 1e-9:
            self.scan_pcd.scale(scale, center=np.zeros(3))
            if self.scan_mesh is not None:
                self.scan_mesh.scale(scale, center=np.zeros(3))

        # Выравнивание основания: низ реконструкции совместить с низом CAD (углы не уходят ниже)
        scan_pts = np.asarray(self.scan_pcd.points)
        cad_min_z = float(np.min(cad_verts[:, 2]))
        scan_min_z = float(np.min(scan_pts[:, 2]))
        shift_z = cad_min_z - scan_min_z
        if abs(shift_z) > 1e-9:
            self.scan_pcd.translate((0, 0, shift_z))
            if self.scan_mesh is not None:
                self.scan_mesh.translate((0, 0, shift_z))
            logging.info(f"Основание выровнено по Z: сдвиг {shift_z:.4f}")

    def _make_cad_pcd(self):
        cad_pcd = o3d.geometry.PointCloud()
        cad_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.cad_mesh.vertices))
        cad_pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.cad_mesh.vertex_normals))
        return cad_pcd

    def _global_registration(self, voxel_size=None, max_correspondence_distance_coarse=None):
        """Грубое совмещение RANSAC (FPFH), чтобы точки попали в область CAD перед ICP."""
        cad_pcd = self._make_cad_pcd()
        cad_pts = np.asarray(cad_pcd.points)
        cad_extent = np.ptp(cad_pts, axis=0)
        if voxel_size is None:
            voxel_size = max(0.01, float(np.mean(cad_extent)) * 0.03)
        if max_correspondence_distance_coarse is None:
            max_correspondence_distance_coarse = float(np.median(np.linalg.norm(cad_pts - np.mean(cad_pts, axis=0), axis=1))) * 2.0
        scan_down = self.scan_pcd.voxel_down_sample(voxel_size)
        cad_down = cad_pcd.voxel_down_sample(voxel_size)
        if len(scan_down.points) < 10 or len(cad_down.points) < 10:
            return np.eye(4)
        scan_down.estimate_normals()
        cad_down.estimate_normals()
        radius_feature = voxel_size * 2
        scan_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            scan_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        cad_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            cad_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            scan_down,
            cad_down,
            scan_fpfh,
            cad_fpfh,
            mutual_filter=True,
            max_correspondence_distance=max_correspondence_distance_coarse,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        n_inliers = len(result.correspondence_set) if hasattr(result, "correspondence_set") else 0
        logging.info(f"Глобальная регистрация (RANSAC): fitness={result.fitness:.4f}, inliers={n_inliers}")
        return result.transformation

    def align_meshes_icp(self, max_iterations=100, threshold=None, use_global_first=True, voxel_global=None):
        """Совмещение: RANSAC (опционально), затем каскад ICP с порогами от размера модели."""
        cad_pcd = self._make_cad_pcd()
        cad_pts_icp = np.asarray(cad_pcd.points)
        cad_extent = np.ptp(cad_pts_icp, axis=0)
        max_extent = float(np.max(cad_extent))
        if max_extent < 1e-9:
            return np.eye(4)
        if use_global_first:
            T_global = self._global_registration(voxel_size=voxel_global)
            self.scan_pcd.transform(T_global)
            if self.scan_mesh is not None:
                self.scan_mesh.transform(T_global)
        # Каскад ICP: пороги от размера модели (чтобы всегда были соответствия)
        if threshold is not None and threshold > 0:
            icp_distances = [max_extent * 0.15, max(threshold, max_extent * 0.002)]
        else:
            icp_distances = [
                max_extent * 0.15,   # грубый
                max_extent * 0.04,   # средний
                max_extent * 0.008,  # точный
            ]
        for i, max_dist in enumerate(icp_distances):
            logging.info("Совмещение ICP (шаг %d, max_dist=%.4f)...", i + 1, max_dist)
            reg = o3d.pipelines.registration.registration_icp(
                self.scan_pcd,
                cad_pcd,
                max_dist,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
            )
            self.scan_pcd.transform(reg.transformation)
            if self.scan_mesh is not None:
                self.scan_mesh.transform(reg.transformation)
            logging.info("  fitness=%.4f, RMSE=%.6f", reg.fitness, reg.inlier_rmse)

        # Дополнительное выравнивание по основанию после ICP:
        # совмещаем минимальный Z реконструкции с минимальным Z CAD,
        # чтобы облако точек доходило до низа модели.
        scan_pts_icp = np.asarray(self.scan_pcd.points)
        cad_min_z_icp = float(np.min(cad_pts_icp[:, 2]))
        scan_min_z_icp = float(np.min(scan_pts_icp[:, 2]))
        shift_z_icp = cad_min_z_icp - scan_min_z_icp
        if abs(shift_z_icp) > 1e-9:
            self.scan_pcd.translate((0, 0, shift_z_icp))
            if self.scan_mesh is not None:
                self.scan_mesh.translate((0, 0, shift_z_icp))
            logging.info(f"Основание выровнено по Z после ICP: доп. сдвиг {shift_z_icp:.4f}")

        return reg.transformation

    def compute_deviations(self, cad_sample_points=100000):
        """Отклонения: для каждой точки реконструкции — знаковое расстояние до поверхности CAD."""
        logging.info("Вычисление отклонений (точки реконструкции -> поверхность CAD)...")
        # Плотная выборка поверхности CAD с нормалями для приближения точка–поверхность
        cad_dense = self.cad_mesh.sample_points_uniformly(
            number_of_points=min(cad_sample_points, 100 * len(self.cad_mesh.vertices))
        )
        cad_dense.estimate_normals()
        cad_pts = np.asarray(cad_dense.points)
        cad_normals = np.asarray(cad_dense.normals)
        scan_pts = np.asarray(self.scan_pcd.points)

        tree = KDTree(cad_pts)
        dists, idxs = tree.query(scan_pts, k=1)
        nearest = cad_pts[idxs]
        normals = cad_normals[idxs]
        vectors = scan_pts - nearest
        signed_distances = np.sum(vectors * normals, axis=1)
        self.deviations = signed_distances
        return self.deviations

    def analyze_deviations(self, tolerance=0.1):
        d = self.deviations

        logging.info("=" * 50)
        logging.info("АНАЛИЗ ОТКЛОНЕНИЙ")
        logging.info("=" * 50)

        logging.info(f"Max: {np.max(d):.6f}")
        logging.info(f"Min: {np.min(d):.6f}")
        logging.info(f"Mean: {np.mean(d):.6f}")
        logging.info(f"Std: {np.std(d):.6f}")
        logging.info(f"RMS: {np.sqrt(np.mean(d ** 2)):.6f}")

        in_tol = np.sum(np.abs(d) <= tolerance) / len(d) * 100
        logging.info(f"В допуске ±{tolerance}: {in_tol:.2f}%")

        plt.figure(figsize=(10, 6))
        plt.hist(d, bins=50)
        plt.axvline(0, linestyle='--')
        plt.axvline(tolerance, linestyle=':')
        plt.axvline(-tolerance, linestyle=':')
        plt.grid(True)
        plt.show()

    def visualize_comparison(self):
        """CAD — прозрачный меш; реконструкция — облако точек, окрашенное по отклонению."""
        deviations = self.deviations
        norm = plt.Normalize(vmin=np.min(deviations), vmax=np.max(deviations))
        cmap = plt.cm.coolwarm
        colors = cmap(norm(deviations))[:, :3]
        self.scan_pcd.colors = o3d.utility.Vector3dVector(colors)

        cad_material = o3d.visualization.rendering.MaterialRecord()
        cad_material.shader = "defaultLitTransparency"
        cad_material.base_color = [0.7, 0.7, 0.7, 0.3]
        scan_material = o3d.visualization.rendering.MaterialRecord()
        scan_material.shader = "defaultLit"
        scan_material.base_color = [1, 1, 1, 1]

        logging.info("Прозрачный: CAD. Цветные точки: реконструкция (отклонение от CAD).")

        geometries = [
            {"name": "CAD", "geometry": self.cad_mesh, "material": cad_material},
            {"name": "Scan", "geometry": self.scan_pcd, "material": scan_material},
        ]
        o3d.visualization.draw(
            geometries,
            title="Сравнение: точки реконструкции и CAD",
            width=1200,
            height=800,
        )

    def export_report(self, output_path="comparison_report.txt"):
        d = self.deviations

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Отчет сравнения 3D моделей\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Время анализа: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Количество точек: {len(d)}\n")
            f.write(f"Max: {np.max(d):.6f}\n")
            f.write(f"Min: {np.min(d):.6f}\n")
            f.write(f"Mean: {np.mean(d):.6f}\n")
            f.write(f"Std: {np.std(d):.6f}\n")
            f.write(f"RMS: {np.sqrt(np.mean(d ** 2)):.6f}\n")

        logging.info(f"Отчет сохранен в {output_path}")


def run_comparison(
    cad_path,
    scan_path=None,
    output_dir=None,
    cad_simplify=50000,
    icp_iterations=50,
    tolerance=0.05,
    no_visualize=False,
    no_histogram=False,
    scale_mode="fit",
):
    """Запуск сравнения CAD и скана. Используется из main.py и из CLI.
    scale_mode: 'fit' — вписать в габариты CAD (основание не уходит ниже), 'median' — по медиане радиусов.
    """
    if output_dir is None:
        output_dir = get_reconstruction_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scan_path is None:
        scan_path_effective = str(get_reconstruction_dir() / "stereo_points.npy")
    else:
        scan_path_effective = scan_path

    start_time = time.time()
    comparator = MeshComparator(cad_path=cad_path, scan_path=scan_path_effective)
    comparator.preprocess_meshes(cad_simplify=cad_simplify, scale_mode=scale_mode)
    comparator.align_meshes_icp(max_iterations=icp_iterations)
    comparator.compute_deviations()
    elapsed = time.time() - start_time
    logging.info(f"Время работы: {elapsed:.2f} с")

    if not no_histogram:
        comparator.analyze_deviations(tolerance=tolerance)

    report_path = output_dir / "comparison_report.txt"
    comparator.export_report(output_path=str(report_path))
    aligned_path = output_dir / "aligned_scan.ply"
    if comparator.scan_mesh is not None:
        o3d.io.write_triangle_mesh(str(aligned_path), comparator.scan_mesh)
    else:
        o3d.io.write_point_cloud(str(aligned_path), comparator.scan_pcd)
    logging.info(f"Выровненная реконструкция сохранена: {aligned_path}")

    if not no_visualize:
        comparator.visualize_comparison()

    return comparator


if __name__ == "__main__":
    default_cad = get_default_cad_path()
    get_cad_models_dir().mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Сравнение 3D: CAD-модель и скан (реконструкция или меш)."
    )
    parser.add_argument(
        "cad",
        type=str,
        nargs="?",
        default=None,
        help=f"Путь к CAD-модели (STL/OBJ и др.). По умолчанию: {default_cad}",
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="Путь к скану (NPY/STL/OBJ). По умолчанию: outputs/<object>/reconstruction/stereo_points.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Папка для отчёта и aligned_scan.ply (по умолчанию: reconstruction)",
    )
    parser.add_argument(
        "--cad-simplify",
        type=int,
        default=50000,
        help="Макс. треугольников CAD после упрощения (0 = не упрощать)",
    )
    parser.add_argument(
        "--icp-iterations",
        type=int,
        default=50,
        help="Число итераций ICP",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Допуск ± для анализа отклонений (мм)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Не открывать 3D визуализацию",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Не показывать гистограмму отклонений",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        choices=("fit", "median"),
        default="fit",
        help="Масштаб: fit — вписать в габариты CAD (по умол.), median — по медиане расстояний",
    )
    args = parser.parse_args()
    cad_path = args.cad or str(default_cad)
    if not os.path.exists(cad_path):
        parser.error(
            f"CAD-модель не найдена: {cad_path}\n"
            f"Положите эталонную модель в {get_cad_models_dir()} (например reference.stl) или укажите путь: comparer.py path/to/model.stl"
        )

    run_comparison(
        cad_path=cad_path,
        scan_path=args.scan,
        output_dir=args.output_dir,
        cad_simplify=args.cad_simplify or None,
        icp_iterations=args.icp_iterations,
        tolerance=args.tolerance,
        no_visualize=args.no_visualize,
        no_histogram=args.no_histogram,
        scale_mode=args.scale_mode,
    )
