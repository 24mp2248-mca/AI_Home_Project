/*********************************
 * BASIC THREE.JS SETUP
 *********************************/

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f2f2);

// Camera
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(60, 60, 60);
camera.lookAt(0, 0, 0);

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Orbit Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enableZoom = true;
controls.enableRotate = true;
controls.enablePan = true;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(50, 100, 50);
scene.add(dirLight);

// Grid helper (for orientation)
scene.add(new THREE.GridHelper(200, 20));

/*********************************
 * CLEAR PREVIOUS ROOMS
 *********************************/
function clearSceneRooms() {
  const objectsToRemove = [];
  scene.traverse(obj => {
    if (obj.userData.isRoom) {
      objectsToRemove.push(obj);
    }
  });
  objectsToRemove.forEach(obj => scene.remove(obj));
}

/*********************************
 * CREATE ROOMS FROM BACKEND JSON
 *********************************/
function createRoomsFromJSON(rooms) {
  clearSceneRooms();

  rooms.forEach(room => {
    const length = room.dimensions.length;
    const width = room.dimensions.width;
    const height = room.dimensions.height;

    const geometry = new THREE.BoxGeometry(length, height, width);

    const material = new THREE.MeshStandardMaterial({
      color: 0xb0b0b0,
      transparent: true,
      opacity: 0.85
    });

    const mesh = new THREE.Mesh(geometry, material);

    mesh.position.set(
      room.position.x,
      height / 2,
      room.position.y
    );

    mesh.userData.isRoom = true;
    scene.add(mesh);
  });
}

/*********************************
 * FETCH DATA FROM BACKEND
 *********************************/
async function uploadSketch(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/upload-sketch/", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  console.log("Backend AI Output:", data);

  if (data.rooms && data.rooms.length > 0) {
    createRoomsFromJSON(data.rooms);
  } else {
    alert("No rooms detected from sketch");
  }
}

/*********************************
 * HANDLE FILE INPUT
 *********************************/
window.handleUpload = function () {
  const input = document.getElementById("sketchInput");
  if (!input.files.length) {
    alert("Please select a sketch image");
    return;
  }
  uploadSketch(input.files[0]);
};

/*********************************
 * ANIMATION LOOP
 *********************************/
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

/*********************************
 * HANDLE WINDOW RESIZE
 *********************************/
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
