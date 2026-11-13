import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export interface CellData {
  soh_pred: number;
  soh_true: number;
  temperature_c: number;
  resistance_ohm: number;
  soh_deviation: number;
}

interface Canvas3DProps {
  packConfig: string; // '4S', '6S', '8S', '12S'
  cells: Record<string, CellData>;
  selectedCell: string | null;
  onSelectCell: (cellId: string | null) => void;
}

// Convert SOH percentage to realistic thermal/health degradation color
function getHealthColor(soh: number): THREE.Color {
  // SOH >= 0.92: Emerald Green (#10b981)
  // SOH ~ 0.85: Yellow-Lime (#eab308)
  // SOH ~ 0.78: Orange (#f97316)
  // SOH <= 0.70: Deep Red (#ef4444)
  if (soh >= 0.90) {
    const t = Math.min(1.0, (soh - 0.90) / 0.10);
    return new THREE.Color().lerpColors(new THREE.Color(0x22c55e), new THREE.Color(0x10b981), t);
  } else if (soh >= 0.80) {
    const t = (soh - 0.80) / 0.10;
    return new THREE.Color().lerpColors(new THREE.Color(0xeab308), new THREE.Color(0x22c55e), t);
  } else if (soh >= 0.72) {
    const t = (soh - 0.72) / 0.08;
    return new THREE.Color().lerpColors(new THREE.Color(0xf97316), new THREE.Color(0xeab308), t);
  } else {
    const t = Math.max(0.0, (soh - 0.50) / 0.22);
    return new THREE.Color().lerpColors(new THREE.Color(0xef4444), new THREE.Color(0xf97316), t);
  }
}

export const Canvas3D: React.FC<Canvas3DProps> = ({
  packConfig,
  cells,
  selectedCell,
  onSelectCell
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cellMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // 1. Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x070b13);
    scene.fog = new THREE.FogExp2(0x070b13, 0.04);

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 5.5, 9.5);
    camera.lookAt(0, 0, 0);

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x60a5fa, 1.5);
    dirLight1.position.set(5, 10, 7);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x10b981, 0.8);
    dirLight2.position.set(-6, -4, -5);
    scene.add(dirLight2);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(20, 20, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -1.2;
    scene.add(gridHelper);

    // 5. Build Pack Geometry
    const cellIds = Object.keys(cells);
    const numCells = Math.max(4, cellIds.length);
    const cellRadius = 0.55;
    const cellHeight = 2.0;
    const spacing = 1.4;

    const cellGroup = new THREE.Group();
    scene.add(cellGroup);

    // Cylindrical cell geometry (18650/21700 format)
    const cylinderGeo = new THREE.CylinderGeometry(cellRadius, cellRadius, cellHeight, 32);
    const capGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.15, 24);
    const capMat = new THREE.MeshStandardMaterial({
      color: 0xd1d5db,
      metalness: 0.9,
      roughness: 0.2
    });

    const startX = -((numCells - 1) * spacing) / 2;

    cellMeshesRef.current.clear();
    cellIds.forEach((id, index) => {
      const posX = startX + index * spacing;
      const cellData = cells[id] || { soh_pred: 0.9 };
      const color = getHealthColor(cellData.soh_pred);

      const cellMat = new THREE.MeshStandardMaterial({
        color: color,
        metalness: 0.35,
        roughness: 0.4,
        emissive: color.clone().multiplyScalar(0.15)
      });

      const cellMesh = new THREE.Mesh(cylinderGeo, cellMat);
      cellMesh.position.set(posX, 0, 0);
      cellMesh.castShadow = true;
      cellMesh.receiveShadow = true;
      cellMesh.userData = { cellId: id };

      // Positive terminal nipple
      const capMesh = new THREE.Mesh(capGeo, capMat);
      capMesh.position.set(0, cellHeight / 2 + 0.075, 0);
      cellMesh.add(capMesh);

      cellGroup.add(cellMesh);
      cellMeshesRef.current.set(id, cellMesh);
    });

    // Nickel strip busbars connecting consecutive cells
    for (let i = 0; i < numCells - 1; i++) {
      const x1 = startX + i * spacing;
      const x2 = startX + (i + 1) * spacing;
      const busbarGeo = new THREE.BoxGeometry(spacing, 0.04, 0.3);
      const busbarMat = new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        metalness: 0.85,
        roughness: 0.25
      });
      const busbar = new THREE.Mesh(busbarGeo, busbarMat);
      busbar.position.set((x1 + x2) / 2, cellHeight / 2 + 0.15, 0);
      cellGroup.add(busbar);
    }

    // Battery Pack Casing Bracket (Transparent acrylic frame)
    const packWidth = numCells * spacing + 0.6;
    const packBoxGeo = new THREE.BoxGeometry(packWidth, cellHeight + 0.6, 2.0);
    const packBoxMat = new THREE.MeshPhysicalMaterial({
      color: 0x334155,
      transparent: true,
      opacity: 0.18,
      roughness: 0.1,
      metalness: 0.1,
      transmission: 0.8,
      thickness: 0.5
    });
    const packBox = new THREE.Mesh(packBoxGeo, packBoxMat);
    packBox.position.set(0, 0.05, 0);
    cellGroup.add(packBox);

    // Mouse Interaction / Orbit Dragging
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let rotationSpeedX = 0.005;
    let rotationSpeedY = 0.005;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      cellGroup.rotation.y += deltaX * rotationSpeedX;
      cellGroup.rotation.x = Math.max(-0.6, Math.min(0.8, cellGroup.rotation.x + deltaY * rotationSpeedY));
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    // Raycaster for cell picking
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const meshes = Array.from(cellMeshesRef.current.values());
      const intersects = raycaster.intersectObjects(meshes, false);

      if (intersects.length > 0) {
        const hitId = intersects[0].object.userData.cellId;
        onSelectCell(hitId);
      } else {
        onSelectCell(null);
      }
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('click', onClick);

    // Resize handler
    const onResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', onResize);

    // Animation loop
    let animId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Subtle resting idle float
      if (!isDragging) {
        cellGroup.position.y = Math.sin(elapsedTime * 0.8) * 0.05;
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('click', onClick);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      if (container.contains(dom)) {
        container.removeChild(dom);
      }
    };
  }, [packConfig, Object.keys(cells).length]);

  // Update cell colors and selection highlight dynamically
  useEffect(() => {
    Object.entries(cells).forEach(([id, data]) => {
      const mesh = cellMeshesRef.current.get(id);
      if (mesh) {
        const mat = mesh.material as THREE.MeshStandardMaterial;
        const targetColor = getHealthColor(data.soh_pred);
        mat.color.copy(targetColor);

        // Highlight selected cell with pulsing emission
        if (selectedCell === id) {
          mat.emissive.setHex(0x38bdf8);
          mesh.scale.set(1.1, 1.05, 1.1);
        } else {
          mat.emissive.copy(targetColor).multiplyScalar(0.15);
          mesh.scale.set(1.0, 1.0, 1.0);
        }
      }
    });
  }, [cells, selectedCell]);

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl">
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      
      {/* 3D View Controls HUD */}
      <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-700/60 text-xs flex items-center space-x-3 pointer-events-none">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
        <span className="font-semibold text-slate-200">Pack: {packConfig}</span>
        <span className="text-slate-400">|</span>
        <span className="text-slate-300">Drag to Orbit, Click Cell to Inspect</span>
      </div>

      {/* Dynamic SOH Color Legend */}
      <div className="absolute bottom-4 right-4 bg-slate-900/80 backdrop-blur-md p-3 rounded-xl border border-slate-700/60 text-xs space-y-1.5 pointer-events-none">
        <div className="font-semibold text-slate-300 mb-1">State of Health (SOH)</div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-emerald-500" />
          <span className="text-slate-300">&gt; 90% (Nominal)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-yellow-500" />
          <span className="text-slate-300">80% - 90% (Mild Fade)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-orange-500" />
          <span className="text-slate-300">72% - 80% (Imbalance Risk)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-red-500" />
          <span className="text-slate-300">&lt; 72% (End of Life / Replacement)</span>
        </div>
      </div>
    </div>
  );
};
