import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// settings 
const LAST_FRAME = 500;                 // Blender timeline: 0-500, 100 frames per page
const CAM_NAMES  = ['CAM_01_Room', 'CAM_02_Music', 'CAM_03_Drop', 'CAM_04_Slide', 'CAM_05_Footer'];
const CAM_AIM    = [0.5, 0.5, -1.4, 1.2, -1.9];   // where each camera looks, as height above the robot's centre
const H_FOV      = 36.2;                // horizontal field of view, same as the Blender preview camera

// which face is showing
const FACES = [
  [0, 'Happy'], [36, 'Wink'], [60, 'Mischief'], [86, 'Happy'],
  [100, 'Surprised'], [131, 'Confused'], [150, 'Happy'], [172, 'Cool'],
  [204, 'Surprised'], [212, 'Shocked'], [268, 'Hurt'],
  [300, 'Sad'], [334, 'Surprised'], [367, 'Shocked'],
  [400, 'Confused'], [485, 'Happy'],
];
const COLORS = {
  Happy: [0.0, 0.2, 1.0], Wink: [0.42, 0.1, 1.0], Mischief: [0.5, 0.15, 1.0],
  Surprised: [1.0, 0.72, 0.04], Confused: [0.05, 0.9, 0.75], Cool: [0.05, 1.0, 0.25],
  Shocked: [0.05, 0.4, 1.0], Hurt: [1.0, 0.04, 0.02], Sad: [0.04, 0.7, 0.85],
};

// renderer, scene, camera
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const isPhone = window.matchMedia('(max-width: 800px)').matches;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, isPhone ? 1.5 : 2));

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;   // soft studio reflections for the metal
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(-3, 6, 6);
scene.add(key);

const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // keep the horizontal view fixed so the robot is the same width on phones and desktops
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(H_FOV / 2)) / camera.aspect));
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// scroll ->t targetFrame = 0, frame = 0;
function readScroll() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  targetFrame = max > 0 ? (window.scrollY / max) * LAST_FRAME : 0;
}
window.addEventListener('scroll', readScroll, { passive: true });
readScroll();

// load the model
const draco = new DRACOLoader();
draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
const loader = new GLTFLoader();
loader.setDRACOLoader(draco);

let ready = false;
let scrubMixer, loopMixer, journeyDuration = 1;
let robot, camNodes = [], headphones;
let hover, vibe, faceIdle, hpOn, hpOff;
let coverMat, glowMats = [];
let headphonesOn = false;

loader.load('assets/robot.glb', (gltf) => {
  const root = gltf.scene;
  scene.add(root);
  console.log('Clips in file:', gltf.animations.map((c) => c.name));

  robot      = root.getObjectByName('ROBOT_Travel');
  headphones = root.getObjectByName('ROBOT_Headphones');
  camNodes   = CAM_NAMES.map((n) => root.getObjectByName(n));
  if (headphones) headphones.visible = false;

  // white cover: pure white, fades in over the room
  coverMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, toneMapped: false, depthWrite: false });
  ['COVER_Backdrop', 'COVER_Floor'].forEach((n) => {
    const o = root.getObjectByName(n);
    if (o) { o.material = coverMat; o.renderOrder = -1; }
  });

  // glowing parts (LEDs and face) - colour is driven from here
  root.traverse((o) => {
    if (o.isMesh && o.material && /M_Emissive_(Blue|Face)/.test(o.material.name) && !glowMats.includes(o.material)) {
      o.material.userData.baseIntensity = o.material.emissiveIntensity;
      glowMats.push(o.material);
    }
  });

  const clip = (name) => THREE.AnimationClip.findByName(gltf.animations, name);

  // 1) scroll-driven clips
  scrubMixer = new THREE.AnimationMixer(root);
  ['Journey', 'Journey_Face'].forEach((n) => { const c = clip(n); if (c) scrubMixer.clipAction(c).play(); });
  journeyDuration = clip('Journey') ? clip('Journey').duration : 1;

  // 2) clips that run on their own clock
  loopMixer = new THREE.AnimationMixer(root);
  const loop = (n) => { const c = clip(n); if (!c) return null; const a = loopMixer.clipAction(c); a.setEffectiveWeight(0); a.play(); return a; };
  const once = (n) => { const c = clip(n); if (!c) return null; const a = loopMixer.clipAction(c); a.setLoop(THREE.LoopOnce); a.clampWhenFinished = true; return a; };
  hover    = loop('Auto_Hover');
  vibe     = loop('Auto_Vibe');
  faceIdle = loop('Auto_Face_Idle');
  hpOn     = once('Auto_Headphones_On');
  hpOff    = once('Auto_Headphones_Off');

  document.getElementById('loader').classList.add('done');
  ready = true;
}, undefined, (err) => console.error('Model failed to load:', err));

//helpers
const smooth = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
const fade = (action, target, dt) => {
  if (!action) return;
  const w = action.getEffectiveWeight();
  action.setEffectiveWeight(w + (target - w) * Math.min(1, dt * 4));
};
const faceAt = (f) => { let name = FACES[0][1]; for (const [start, n] of FACES) if (f >= start) name = n; return name; };

const vA = new THREE.Vector3(), vB = new THREE.Vector3(), robotPos = new THREE.Vector3();
const camPos = new THREE.Vector3(), lookAt = new THREE.Vector3(), lookNow = new THREE.Vector3();
const glow = new THREE.Color(0, 0.2, 1), glowTarget = new THREE.Color();
let firstFrame = true;

/* ---------- every frame ---------- */
const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.1);
  const time = clock.elapsedTime;
  if (!ready) return;

  // smooth the scroll so scrubbing never jumps
  frame += (targetFrame - frame) * (1 - Math.exp(-dt * 6));

  // scroll-driven animation
  scrubMixer.setTime(Math.min(frame / LAST_FRAME, 0.9999) * journeyDuration);

  // self-running animation
  fade(hover,    frame < 150 ? 1 : 0, dt);                 // page 1, and while he comes down
  fade(vibe,     frame >= 150 && frame < 204 ? 1 : 0, dt); // page 2, until the doors unlatch
  fade(faceIdle, frame >= 200 ? 1 : 0, dt);                // pages 3-5

  const wantHeadphones = frame >= 140 && frame < 196;
  if (wantHeadphones !== headphonesOn && hpOn && hpOff) {
    headphonesOn = wantHeadphones;
    headphones.visible = true;
    if (wantHeadphones) { hpOff.stop(); hpOn.reset().play(); }
    else                { hpOn.stop();  hpOff.reset().play(); }
  }
  loopMixer.update(dt);

  // white cover fades in over frames 100-130
  if (coverMat) coverMat.opacity = Math.min(1, Math.max(0, (frame - 100) / 30));

  // glow colour follows the mood, and breathes a little on pages 3-5
  glowTarget.setRGB(...COLORS[faceAt(frame)]);
  glow.lerp(glowTarget, Math.min(1, dt * 8));
  const breathe = frame >= 200 ? 1 + 0.22 * Math.sin((time * Math.PI * 2) / 2.5) : 1;
  glowMats.forEach((m) => { m.emissive.copy(glow); m.emissiveIntensity = m.userData.baseIntensity * breathe; });

  // camera: blend between the markers, which travel with the robot
  if (robot && camNodes.every(Boolean)) {
    const i = Math.min(4, Math.floor(frame / 100));
    const t = smooth((frame - 100 * i) / 50);
    const a = Math.max(0, i - 1);
    camNodes[a].getWorldPosition(vA);
    camNodes[i].getWorldPosition(vB);
    camPos.lerpVectors(vA, vB, t);
    robot.getWorldPosition(robotPos);
    const aim = CAM_AIM[a] + (CAM_AIM[i] - CAM_AIM[a]) * t;
    lookAt.set(camPos.x, robotPos.y + aim, robotPos.z);

    const k = firstFrame ? 1 : 1 - Math.exp(-dt * 5);   // soft follow, so a hard landing never jerks the view
    camera.position.lerp(camPos, k);
    lookNow.lerp(lookAt, k);
    camera.lookAt(lookNow);
    firstFrame = false;
  }

  renderer.render(scene, camera);
}
tick();
