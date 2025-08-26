const WS_URL    = "ws://127.0.0.1:8765";
const THRESHOLD = 0.70;
const USE_CANVAS = true; // set false để test hiển thị trực tiếp <video>

const $messages    = document.getElementById('messages');
const $video       = document.getElementById('video');
const $canvas      = document.getElementById('canvas');
const $videoDot    = document.getElementById('video-dot');
const $badgeVideo  = document.getElementById('badge-video');
const $btnWS       = document.getElementById('btn-ws');
const $btnDemo     = document.getElementById('btn-demo');
const $hint        = document.getElementById('hint');
const $toggleCam   = document.getElementById('toggleCamBtn');
const $uploadDot   = document.getElementById('upload-dot');


//Nút setting----------------------------------------------------------------------------------------------------------------
// Nút điều khiển menu Setting
const $toggleSetting = document.getElementById('toggle-setting');
const $settingOptions = document.getElementById('setting-options');

// Các tùy chọn trong menu Setting
const $optionUploadVideo = document.getElementById('option-upload-video');
const $optionShowKeypoints = document.getElementById('option-show-keypoints');

// Phần Upload Video và Display Keypoints
const $videoUploadSection = document.getElementById('video-upload-section');
const $keypointsDisplay = document.getElementById('keypoints-display');

// Toggle hiển thị menu Setting
$toggleSetting.addEventListener('click', () => {
    $settingOptions.classList.toggle('hidden'); // Hiện/ẩn menu tùy chọn
});


$optionUploadVideo.addEventListener('click', () => {
    // Toggle hidden
    $videoUploadSection.classList.toggle('hidden');

    // Kiểm tra đang bật hay không để áp dụng màu dot
    const isOn = !$videoUploadSection.classList.contains('hidden');

    if (isOn) {
        $uploadDot.classList.add('on');   // Màu đỏ
    } 
    else {
        $uploadDot.classList.remove('on'); // Màu xanh
    }

    // Sau khi chọn xong, tự động ẩn menu
    $settingOptions.classList.add('hidden');
});

// Trạng thái: bật/tắt hiển thị keypoints
let isKeypointsOn = false;

// Xử lý khi chọn "Show Keypoints"
$optionShowKeypoints.addEventListener('click', () => {
    isKeypointsOn = !isKeypointsOn; // Đảo trạng thái bật/tắt

    if (isKeypointsOn) {
        $optionShowKeypoints.textContent = 'Show Keypoints (On)';
        canvasElement.style.display = 'block'; // Hiện canvas
    } 
    else {
        $optionShowKeypoints.textContent = 'Show Keypoints (Off)';
        canvasElement.style.display = 'none'; // Ẩn canvas
    }
});


//Nút điều khiển camera----------------------------------------------------------------------------------------------------------------------
let camStream = null, rafId = null, videoActive = false;

// set canvas size khi có metadata
$video.addEventListener('loadedmetadata', () => {
    if ($canvas) {
    $canvas.width  = $video.videoWidth  || 640;
    $canvas.height = $video.videoHeight || 480;
    }
});

let hands;   // Tạo biến lưu instance của Mediapipe
let ctx;     // Canvas Context để vẽ hình
let latestHandLandmarks = null;

async function initializeMediapipe() {
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 2,                   // Số lượng bàn tay tối đa nhận diện
        modelComplexity: 1,               // Độ phức tạp của model (Default: 1)
        minDetectionConfidence: 0.7,      // Độ tin cậy tối thiểu của việc phát hiện bàn tay
        minTrackingConfidence: 0.5,       // Độ tin cậy tối thiểu của việc tracking keypoint
    });

    // Khi có kết quả từ model Mediapipe
    hands.onResults((results) => {
        if (!$canvas) return;

        // Xóa canvas trước khi vẽ lại
        ctx.clearRect(0, 0, $canvas.width, $canvas.height);
        // Vẽ khung video + keypoints
        ctx.drawImage(results.image, 0, 0, $canvas.width, $canvas.height);

        // Vẽ keypoints (dùng drawing_utils từ Mediapipe)
        if (results.multiHandLandmarks && isKeypointsOn) {
            latestHandLandmarks = results.multiHandLandmarks;

            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2,
                });
                drawLandmarks(ctx, landmarks, {
                    color: '#FF0000',
                    lineWidth: 1,
                    radius: 5,
                });
            }
        }
        else {
            latestHandLandmarks = null;
        }
    });
}

async function startCam() {
    // Khởi tạo Mediapipe nếu chưa có
    if (!hands) {
        await initializeMediapipe();
    }

    // Mở camera
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    $video.srcObject = camStream;
    ctx = $canvas.getContext('2d'); // Khởi tạo context cho canvas
    await $video.play();

    // Chạy Mediapipe nhận diện bàn tay
    const updateHands = async () => {
        if ($video.readyState >= 2 && $canvas && $canvas.width && $canvas.height) {
            await hands.send({ image: $video });
        }
        rafId = requestAnimationFrame(updateHands); // Lặp lại
    };
    rafId = requestAnimationFrame(updateHands);

    videoActive = true; 
    updateVideoBadge();
    $toggleCam.textContent = 'Turn off camera';
}

function stopCam() {
    if (camStream) camStream.getVideoTracks().forEach(t => t.stop());
    camStream = null; 
    $video.srcObject = null;
    
    // Chuyển màn hình video sang màu đen
    if ($canvas && USE_CANVAS) {
        const ctx = $canvas.getContext('2d');
        ctx.fillStyle = '#000'; // Màu đen
        ctx.fillRect(0, 0, $canvas.width, $canvas.height); // Tô toàn bộ canvas
    }

    if (rafId) cancelAnimationFrame(rafId), rafId = null;
    videoActive = false; 
    updateVideoBadge();
    $toggleCam.textContent = 'Turn on camera';
}

$toggleCam.addEventListener('click', async () => {
    try { 
        camStream ? stopCam() : await startCam(); 
    }
    catch (e) { 
        console.error(e); alert('Không mở được camera. Dùng HTTPS hoặc http://localhost và cho phép quyền.'); 
    }
});

function updateVideoBadge() {
    if (videoActive && isSendingFrames) {
        $badgeVideo.textContent = 'Playing';
        $badgeVideo.classList.add('ok');
    } 
    else if (videoActive) {
        $badgeVideo.textContent = 'Not playing';
        $badgeVideo.classList.remove('ok');
        $videoDot.style.background = '#22c55e'; 
    } 
    else {
        $badgeVideo.textContent = 'Not playing';
        $badgeVideo.classList.remove('ok');
        $videoDot.style.background = '#ef4444';
    }
}


//Nút bắt đầu gửi keypoint từ camera để cho thủ ngữ có thể nhận diện---------------------------------------------------------------------------------
let isSendingFrames = false;
let sendInterval = null;

const $btnSendFrame = document.getElementById('btn-send-frame');

// Hàm gửi khung hình lên backend qua HTTP
async function sendKeypointsToBackend(keypoints) {
    try {
        const response = await fetch('/upload-keypoints', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({keypoints}), 
            });

        if (response.ok) {
            console.log('keypoints đã được gửi thành công.');
        } 
        else {
            console.error('keypoints không thể gửi khung hình tới backend.');
        }
    } 
    catch (error) {
        console.error('Lỗi khi gửi keypoints:', error);
    }
}

// Hàm bắt đầu gửi keypoint
function startSendingKeyPoint() {
    if (!videoActive) {
        alert("Camera chưa mở! Vui lòng mở camera trước.");
        return;
    }

    sendInterval = setInterval(() => {
        // Kiểm tra nếu không có kết quả nhận diện bảng tay
        if (!latestHandLandmarks) {
            console.warn("Không tìm thấy keypoints nào để gửi.");
            return;
        }

        // Xử lý keypoints từ bàn tay được nhận diện
        const keypoints = latestHandLandmarks.map((landmarks, handIndex) =>
            landmarks.map((point) => ({
                x: point.x, // Tọa độ X (0-1)
                y: point.y, // Tọa độ Y (0-1)
                z: point.z, // Tọa độ Z (chiều sâu)
            }))
        );

        sendKeypointsToBackend(keypoints); // Gửi lên backend
    }, 300); // Gửi mỗi 300ms
}

// Hàm dừng gửi keypoint
function stopSendingKeyPoint() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }
}

// Sự kiện cho nút "Bắt đầu gửi" hoặc "Ngừng gửi"
$btnSendFrame.addEventListener('click', () => {
    if (!isSendingFrames) {
        startSendingKeyPoint();
        isSendingFrames = true;
        $btnSendFrame.textContent = 'Stop send keypoint';
    } 
    else {
        stopSendingKeyPoint();
        isSendingFrames = false;
        $btnSendFrame.textContent = 'Send keypoint';
    }
    updateVideoBadge();
});


//Gửi video lên backend--------------------------------------------------------------------------------------------------------------------------
const $fileInput = document.getElementById('video-file');               // Đầu vào để chọn file
const $btnUploadVideo = document.getElementById('btn-upload-video');    // Nút upload video
const $uploadStatus = document.getElementById('upload-status');         // Thông báo trạng thái tải

// Hàm gửi video lên backend
async function uploadVideo(file) {
    try {
        $uploadStatus.textContent = 'Đang tải video lên...';            // Hiển thị trạng thái

        // Tạo một FormData object để gửi tệp dưới dạng dạng "multipart/form-data"
        const formData = new FormData();
        formData.append('video', file); // Thêm file vào form

        // Gửi tệp tới backend qua API POST
        const response = await fetch('/upload-video', {
        method: 'POST',
        body: formData,
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Kết quả từ backend:', result);
            $uploadStatus.textContent = 'Tải video lên thành công!';
        } 
        else {
            console.error('Lỗi khi tải video:', response.statusText);
            $uploadStatus.textContent = 'Tải video thất bại. Hãy thử lại.';
        }
    } 
    catch (error) {
        console.error('Lỗi khi gửi video:', error);
        $uploadStatus.textContent = 'Tải lên thất bại. Hãy thử lại.';
    }
}

// Gắn sự kiện vào nút khi được nhấn
$btnUploadVideo.addEventListener('click', () => {
    const file = $fileInput.files[0]; // Lấy file đầu vào
    if (!file) {
        alert('Vui lòng chọn một tệp video!');
        return;
    }

    // Chỉ gửi nếu file đúng định dạng
    if (!file.type.startsWith('video/')) {
        alert('Chỉ hỗ trợ tệp video!');
        return;
    }

    uploadVideo(file); // Gọi hàm tải video lên backend
});


//Chat render-----------------------------------------------------------------------------------------------------------------------------------
function pushSystem(text){
    if(!text) return;
    const div = document.createElement('div');
    div.className = 'msg system';
    div.textContent = text;
    $messages.appendChild(div);
    $messages.scrollTop = $messages.scrollHeight;
    if ($hint) $hint.style.display = 'none';
}

let _last='', _lastAt=0;
function onRecognized(payload){
    const text = typeof payload==='string' ? payload : (payload?.text ?? '');
    const conf = typeof payload==='object' ? (payload.confidence ?? payload.score ?? 1) : 1;
    if (!videoActive || !text || conf < THRESHOLD) return;
    const now = Date.now(); if (text===_last && now-_lastAt<800) return;
    _last=text; _lastAt=now;
    pushSystem(text.trim());
}
window.onRecognized = onRecognized;


//Websocket (backend)---------------------------------------------------------------------------------------------------------------------------------
function connectWS(url = WS_URL) {
    const ws = new WebSocket(url);
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'WS…';
    document.querySelector('.actions').appendChild(badge);

    ws.onopen = () => {
        badge.textContent = 'WS connected';
        badge.classList.add('ok');
    };

    ws.onmessage = (e) => {
        let d = e.data;
        try {
        d = JSON.parse(e.data);
        } catch {}
        onRecognized(d);
    };

    ws.onerror = () => {
        badge.textContent = 'WS error';
    };

    ws.onclose = () => {
        badge.textContent = 'WS closed';

        setTimeout(() => {
            badge.remove(); 
        }, 2000); 
    };

    return ws;
}
document.getElementById('btn-ws').addEventListener('click', ()=>connectWS());


//Demo--------------------------------------------------------------------------------------------------------------------------------------------------------------
document.getElementById('btn-demo').addEventListener('click', ()=>{
    if (!videoActive) { videoActive = true; updateVideoBadge(); }
    const demo = ['Demo start','Nhận diện: Xin chào','Chuyển văn bản → chat','Kết thúc'];
    let i=0; const id=setInterval(()=>{ if(i<demo.length) onRecognized({text:demo[i++],confidence:0.95}); else clearInterval(id); },800);
});

// Seed
pushSystem('Panel is ready. Ready to get word from AI');