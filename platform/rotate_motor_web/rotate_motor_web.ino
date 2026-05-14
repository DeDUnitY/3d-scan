#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// Драйвер: TMC2209
// Подключения к ESP8266 (Wemos D1 mini):
//   TMC2209 STEP -> D7 (GPIO13)
//   TMC2209 DIR  -> D8 (GPIO15)
//   TMC2209 EN   -> D0 (GPIO16)  — LOW = мотор включён, HIGH = выключен (active LOW)
//
// Примечание по D0 (GPIO16): на этом пине при старте ESP8266 стоит HIGH,
// что для TMC2209 означает "мотор выключен" — это безопасное состояние при загрузке.
#define STEP_PIN D7
#define DIR_PIN  D8
#define ENABLE_PIN D0
#define LASER_PIN D1   // пин для MOSFET, управляющего лазерной подсветкой
// Для большинства MOSFET-модулей с опторазвязкой вход IN/PWM активен по LOW.
// 1 -> LOW включает нагрузку; 0 -> обычная логика (HIGH включает).
#define LASER_ACTIVE_LOW 0

// Пины микрошагов для TMC2209 (MS1, MS2)
// Таблица режимов TMC2209:
//   MS1=LOW,  MS2=LOW  -> 1/8  микрошага (1600 шагов/оборот)
//   MS1=HIGH, MS2=LOW  -> 1/32 микрошага (6400 шагов/оборот)
//   MS1=LOW,  MS2=HIGH -> 1/64 микрошага (12800 шагов/оборот)
//   MS1=HIGH, MS2=HIGH -> 1/16 микрошага (3200 шагов/оборот)
//
// ВАРИАНТ 1: Управление через ESP8266 (программно) — укажите пины ниже.
// ВАРИАНТ 2: Фиксированное подключение через резисторы/перемычки к GND или VIO (3.3V).
//   Для 1/32: MS1 -> VIO (3.3V), MS2 -> GND
//
// Оставьте -1, если режим задаётся аппаратно (через резисторы/перемычки).
#define MS1_PIN -1
#define MS2_PIN -1
// Если MS1/MS2 не подключены к ESP8266, укажите фактический режим драйвера:
//   0=1/8, 1=1/16, 2=1/32, 3=1/64.
// Для TMC2209 без настройки MS-пинов чаще всего это 1/8.
#define FIXED_MICROSTEP_MODE 3

const char* ssid = "Cudy-EFB4";
const char* password = "MYpass-1";

ESP8266WebServer server(80);

int speedDelay = 2000;   // микросекунды между шагами (рассчитывается автоматически)
float revTime = 1.2f;  // время одного шага в секундах (по умолчанию; меньше = быстрее)
bool direction = true;
bool isRunning = false;  // флаг выполнения последовательности
// Режим микрошагов TMC2209: 0=1/8, 1=1/16, 2=1/32, 3=1/64 (по умолчанию 1/32)
int microstepMode = FIXED_MICROSTEP_MODE;
bool holdEnabled = false;  // true: удерживать мотор включённым между поворотами

// Число шагов (позиций) за полный оборот 360°.
// Может меняться из Web (ползунок "Количество шагов").
int positionsPerTurn = 10;
long currentPosition = 0;

// Состояние лазерной подсветки (через MOSFET на LASER_PIN)
bool laserOn = false;
int laserPwm = 0;  // 0..1023

void setLaser(bool on) {
  laserOn = on;
  laserPwm = on ? 1023 : 0;
  int pwmOut = laserPwm;
  if (LASER_ACTIVE_LOW) {
    pwmOut = 1023 - pwmOut;
  }
  analogWrite(LASER_PIN, pwmOut);
}

void setLaserPwm(int pwm) {
  if (pwm < 0) pwm = 0;
  if (pwm > 1023) pwm = 1023;
  laserPwm = pwm;
  laserOn = (pwm > 0);
  int pwmOut = pwm;
  if (LASER_ACTIVE_LOW) {
    pwmOut = 1023 - pwmOut;
  }
  analogWrite(LASER_PIN, pwmOut);
}

void handleLaser() {
  if (!server.hasArg("pwm")) {
    server.send(200, "text/plain", String(laserPwm));
    return;
  }

  int pwm = server.arg("pwm").toInt();
  if (pwm < 0 || pwm > 1023) {
    server.send(400, "text/plain", "Invalid pwm");
    return;
  }

  setLaserPwm(pwm);
  server.send(200, "text/plain", "OK");
}

// Функция для получения количества шагов на оборот в зависимости от режима (TMC2209)
int getStepsPerRev() {
  switch(microstepMode) {
    case 0: return 1600;   // 1/8  микрошага
    case 1: return 3200;   // 1/16 микрошага
    case 2: return 6400;   // 1/32 микрошага
    case 3: return 12800;  // 1/64 микрошага
    default: return 6400;
  }
}

// Функция для получения количества микрошагов в одном полном шаге
int getMicrostepsPerFullStep() {
  switch(microstepMode) {
    case 0: return 8;
    case 1: return 16;
    case 2: return 32;
    case 3: return 64;
    default: return 32;
  }
}

// Шагов на один шаг стола (одна позиция). Целое число — мотор полностью останавливается.
int getStepsPerFixedAngle() {
  int spr = getStepsPerRev();
  if (spr <= 0 || positionsPerTurn <= 0) return 0;
  return spr / positionsPerTurn;
}

int getStepsPerTurnByCount(int count) {
  int spr = getStepsPerRev();
  if (spr <= 0 || count <= 0) return 0;
  return spr / count;
}

// Точный угол одного поворота в градусах (шаги / шаги_на_оборот * 360)
float getExactAngleDegrees() {
  int spr = getStepsPerRev();
  int steps = getStepsPerFixedAngle();
  if (spr <= 0) return 0.0f;
  return (float)steps * 360.0f / (float)spr;
}

float getExactAngleDegreesByCount(int count) {
  int spr = getStepsPerRev();
  int steps = getStepsPerTurnByCount(count);
  if (spr <= 0) return 0.0f;
  return (float)steps * 360.0f / (float)spr;
}

// Функция для установки режима микрошагов (TMC2209: MS1, MS2)
void setMicrostepMode(int mode) {
  if (MS1_PIN < 0 || MS2_PIN < 0) {
    Serial.println("ВНИМАНИЕ: Пины MS1, MS2 не подключены к ESP8266!");
    Serial.println("Настройте режим микрошагов аппаратно (перемычками MS1/MS2).");
    return;
  }
  
  microstepMode = mode;
  bool ms1, ms2;
  
  // Таблица TMC2209:
  //   MS1=L, MS2=L -> 1/8
  //   MS1=H, MS2=H -> 1/16
  //   MS1=H, MS2=L -> 1/32
  //   MS1=L, MS2=H -> 1/64
  switch(mode) {
    case 0: ms1=0; ms2=0; break;  // 1/8
    case 1: ms1=1; ms2=1; break;  // 1/16
    case 2: ms1=1; ms2=0; break;  // 1/32
    case 3: ms1=0; ms2=1; break;  // 1/64
    default: ms1=1; ms2=0; break; // 1/32
  }
  
  digitalWrite(MS1_PIN, ms1);
  digitalWrite(MS2_PIN, ms2);
  
  Serial.print("Режим микрошагов установлен: ");
  Serial.print(mode);
  Serial.print(" (");
  Serial.print(getStepsPerRev());
  Serial.println(" шагов/оборот)");
}

void stepMotor(int steps) {
  // Сначала устанавливаем направление ДО включения двигателя
  digitalWrite(DIR_PIN, direction);
  delayMicroseconds(10);  // время установки DIR для TMC2209 (достаточно ~20 нс, берём с запасом)
  
  // Включаем двигатель после установки направления
  digitalWrite(ENABLE_PIN, LOW);  // включить двигатель (LOW = включён для TMC2209)
  delay(10);  // небольшая задержка для стабилизации
  
  // Адаптивный интервал обработки WiFi в зависимости от количества шагов
  // Для больших значений шагов (микрошаги) обрабатываем реже для плавности
  int wifiInterval = (steps > 1000) ? 50 : (steps > 500) ? 25 : 10;
  
  unsigned long lastWifiTime = 0;
  const unsigned long wifiIntervalMs = 10;  // обрабатывать WiFi не чаще раз в 10 мс
  
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(speedDelay);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(speedDelay);
    
    // Обработка WiFi с адаптивным интервалом и ограничением по времени
    unsigned long currentTime = millis();
    if (i % wifiInterval == 0 && (currentTime - lastWifiTime) >= wifiIntervalMs) {
      server.handleClient();
      yield();  // Дать время другим задачам ESP8266
      lastWifiTime = currentTime;
      ESP.wdtFeed();  // Сброс watchdog
    }
  }
  
  delay(50);  // задержка после движения
  if (!holdEnabled) {
    digitalWrite(ENABLE_PIN, HIGH);  // выключить двигатель (освободить вал)
  }
}

void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 20px;
  color: #333;
}
.container {
  max-width: 700px;
  margin: 0 auto;
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  padding: 30px;
}
h1 {
  text-align: center;
  color: #667eea;
  margin-bottom: 10px;
  font-size: 28px;
}
.subtitle {
  text-align: center;
  color: #666;
  margin-bottom: 30px;
  font-size: 14px;
}
.settings-group {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid #e9ecef;
}
.settings-group h3 {
  color: #495057;
  margin-bottom: 15px;
  font-size: 18px;
}
.setting-item {
  margin-bottom: 20px;
}
.setting-item:last-child {
  margin-bottom: 0;
}
.setting-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-weight: 600;
  color: #495057;
}
.setting-value {
  font-size: 20px;
  color: #667eea;
  font-weight: bold;
}
input[type="range"] {
  width: 100%;
  height: 8px;
  border-radius: 5px;
  background: #dee2e6;
  outline: none;
  -webkit-appearance: none;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #667eea;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
input[type="range"]::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #667eea;
  cursor: pointer;
  border: none;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.setting-hint {
  font-size: 12px;
  color: #6c757d;
  margin-top: 5px;
}
select {
  width: 100%;
  padding: 12px;
  border: 2px solid #dee2e6;
  border-radius: 8px;
  font-size: 16px;
  background: white;
  color: #495057;
  cursor: pointer;
  transition: border-color 0.3s;
}
select:focus {
  outline: none;
  border-color: #667eea;
}
.button-group {
  display: flex;
  gap: 15px;
  margin: 25px 0;
  flex-wrap: wrap;
}
.btn {
  flex: 1;
  min-width: 200px;
  padding: 15px 25px;
  font-size: 18px;
  font-weight: 600;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.btn:active {
  transform: translateY(0);
}
.btn-primary {
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
}
.btn-danger {
  background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
  color: white;
}
.btn-secondary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 16px;
  padding: 12px 20px;
}
.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.status {
  text-align: center;
  padding: 15px;
  border-radius: 10px;
  font-weight: 600;
  font-size: 16px;
  margin-top: 20px;
  min-height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.status.info {
  background: #e3f2fd;
  color: #1976d2;
}
.status.success {
  background: #e8f5e9;
  color: #388e3c;
}
.status.error {
  background: #ffebee;
  color: #d32f2f;
}
.exact-angle {
  text-align: center;
  margin-bottom: 20px;
  padding: 12px 20px;
  background: #e8eaf6;
  border-radius: 10px;
  color: #3949ab;
  font-size: 16px;
}
.exact-angle strong {
  font-size: 18px;
}
</style>
</head>
<body>
<div class="container">
  <h1>📸 Фотограмметрия</h1>
  <p class="subtitle">Автоматическая съемка с фиксированным шагом</p>
  <p class="exact-angle" id="exactAngle">Один шаг: <strong id="angleValue">—</strong>° (всего позиций за круг: 10)</p>

  <div class="settings-group">
    <h3>⚙️ Настройки последовательности</h3>
    
    <div class="setting-item">
      <div class="setting-label">
        <span>Количество шагов</span>
        <span class="setting-value" id="count">10</span>
      </div>
      <input type="range" min="1" max="120" value="10" id="rotations">
      <div class="setting-hint">10 шагов = полный круг 360° (мотор останавливается в полном шаге)</div>
    </div>

    <div class="setting-item">
      <div class="setting-label">
        <span>Время одного шага (сек)</span>
        <span class="setting-value" id="t">1.2</span>
      </div>
      <input type="range" min="0.3" max="10" value="1.2" step="0.1" id="revTime">
      <div class="setting-hint">Время одного шага (меньше = быстрее поворот)</div>
    </div>

    <div class="setting-item">
      <div class="setting-label">
        <span>Задержка между поворотами (сек)</span>
        <span class="setting-value" id="delay">3.0</span>
      </div>
      <input type="range" min="0" max="30" value="3" step="0.5" id="pauseTime">
      <div class="setting-hint">Время ожидания перед следующим поворотом (для съемки)</div>
    </div>

    <div class="setting-item">
      <div class="setting-label">
        <span>Направление</span>
      </div>
      <select id="direction">
        <option value="1">↻ По часовой стрелке (CW)</option>
        <option value="0">↺ Против часовой стрелки (CCW)</option>
      </select>
    </div>
  </div>

  <div class="settings-group">
    <h3>🔦 Лазер</h3>

    <div class="setting-item">
      <div class="setting-label">
        <span>Мощность лазера (PWM)</span>
        <span class="setting-value" id="laserValue">0</span>
      </div>
      <input type="range" min="0" max="1023" value="0" step="1" id="laserPwm">
      <div class="setting-hint">0 = выключен, 1023 = максимум</div>
    </div>

    <div class="button-group">
      <button onclick="setLaserPwmValue(0)" class="btn btn-secondary">Лазер выкл</button>
      <button onclick="setLaserPwmValue(256)" class="btn btn-secondary">25%</button>
      <button onclick="setLaserPwmValue(512)" class="btn btn-secondary">50%</button>
      <button onclick="setLaserPwmValue(768)" class="btn btn-secondary">75%</button>
      <button onclick="setLaserPwmValue(1023)" class="btn btn-secondary">100%</button>
    </div>
  </div>

  <div class="button-group">
    <button onclick="startSequence()" id="startBtn" class="btn btn-primary">
      ▶ Запустить последовательность
    </button>
    <button onclick="stopSequence()" id="stopBtn" class="btn btn-danger" style="display: none;">
      ⏹ Остановить
    </button>
  </div>

  <div class="button-group">
    <button onclick="sendSingle(1)" class="btn btn-secondary">↻ Один поворот CW</button>
    <button onclick="sendSingle(0)" class="btn btn-secondary">↺ Один поворот CCW</button>
  </div>

  <div id="status" class="status"></div>
</div>

<script>
let isRunning = false;
let laserUpdateTimer = null;

// Сохранение настроек в localStorage
function saveSettings() {
  const settings = {
    rotations: document.getElementById('rotations').value,
    revTime: document.getElementById('revTime').value,
    pauseTime: document.getElementById('pauseTime').value,
    direction: document.getElementById('direction').value
  };
  localStorage.setItem('photogrammetrySettings', JSON.stringify(settings));
}

// Загрузка настроек из localStorage
function loadSettings() {
  const saved = localStorage.getItem('photogrammetrySettings');
  if (saved) {
    try {
      const settings = JSON.parse(saved);
      if (settings.rotations) document.getElementById('rotations').value = settings.rotations;
      if (settings.revTime) document.getElementById('revTime').value = settings.revTime;
      if (settings.pauseTime) document.getElementById('pauseTime').value = settings.pauseTime;
      if (settings.direction) document.getElementById('direction').value = settings.direction;
    } catch(e) {
      console.error('Ошибка загрузки настроек:', e);
    }
  }
  updateDisplay();
}

function updateDisplay() {
  const rotations = document.getElementById('rotations').value;
  const revTime = parseFloat(document.getElementById('revTime').value).toFixed(1);
  const pauseTime = parseFloat(document.getElementById('pauseTime').value).toFixed(1);
  
  document.getElementById('count').innerHTML = rotations;
  document.getElementById('t').innerHTML = revTime;
  document.getElementById('delay').innerHTML = pauseTime;
  
  saveSettings(); // Сохраняем при каждом изменении
}

function updateLaserDisplay() {
  const laserPwm = document.getElementById('laserPwm').value;
  document.getElementById('laserValue').innerHTML = laserPwm;
}

function showStatus(message, type = 'info') {
  const statusEl = document.getElementById('status');
  statusEl.innerHTML = message;
  statusEl.className = 'status ' + type;
}

function startSequence() {
  if (isRunning) return;
  
  let count = parseInt(document.getElementById('rotations').value);
  let revTime = parseFloat(document.getElementById('revTime').value);
  let pauseTime = parseFloat(document.getElementById('pauseTime').value);
  let dir = document.getElementById('direction').value;
  
  document.getElementById('startBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'inline-block';
  showStatus('⏳ Выполняется...', 'info');
  isRunning = true;
  
  fetch(`/sequence?count=${count}&revTime=${revTime}&pauseTime=${pauseTime}&dir=${dir}`)
    .then(() => {
      isRunning = false;
      document.getElementById('startBtn').style.display = 'inline-block';
      document.getElementById('stopBtn').style.display = 'none';
      showStatus('✅ Готово!', 'success');
      setTimeout(() => showStatus('', ''), 3000);
    })
    .catch(() => {
      isRunning = false;
      document.getElementById('startBtn').style.display = 'inline-block';
      document.getElementById('stopBtn').style.display = 'none';
      showStatus('❌ Ошибка!', 'error');
      setTimeout(() => showStatus('', ''), 3000);
    });
}

function stopSequence() {
  fetch('/stop');
  isRunning = false;
  document.getElementById('startBtn').style.display = 'inline-block';
  document.getElementById('stopBtn').style.display = 'none';
  showStatus('⏹ Остановлено', 'info');
  setTimeout(() => showStatus('', ''), 3000);
}

function sendSingle(dir) {
  let t = document.getElementById('revTime').value;
  let count = parseInt(document.getElementById('rotations').value);
  // dir: 1 = CW, 0 = CCW (явно передаём число как строку)
  fetch('/move?revTime=' + encodeURIComponent(t) + '&count=' + encodeURIComponent(count) + '&dir=' + (dir ? '1' : '0'));
}

function sendLaserPwm() {
  const pwm = document.getElementById('laserPwm').value;
  fetch('/laser?pwm=' + encodeURIComponent(pwm))
    .catch(() => showStatus('❌ Ошибка управления лазером', 'error'));
}

function queueLaserPwmUpdate() {
  updateLaserDisplay();
  if (laserUpdateTimer) {
    clearTimeout(laserUpdateTimer);
  }
  laserUpdateTimer = setTimeout(sendLaserPwm, 120);
}

function setLaserPwmValue(pwm) {
  document.getElementById('laserPwm').value = pwm;
  updateLaserDisplay();
  sendLaserPwm();
}

function loadLaserPwm() {
  fetch('/laser')
    .then(r => r.text())
    .then(text => {
      const pwm = parseInt(text, 10);
      if (!isNaN(pwm)) {
        document.getElementById('laserPwm').value = pwm;
        updateLaserDisplay();
      }
    })
    .catch(() => {});
}

// Загрузка точного угла с платы
function loadExactAngle() {
  const rotations = parseInt(document.getElementById('rotations').value);
  fetch('/angle?count=' + encodeURIComponent(rotations))
    .then(r => r.text())
    .then(text => {
      const val = parseFloat(text);
      if (!isNaN(val)) {
        document.getElementById('angleValue').textContent = val.toFixed(4);
      }
    })
    .catch(() => { document.getElementById('angleValue').textContent = '—'; });
}

// Инициализация
document.addEventListener('DOMContentLoaded', function() {
  loadSettings(); // Загружаем сохраненные настройки
  loadExactAngle(); // Показываем точный угол поворота
  loadLaserPwm(); // Показываем текущее значение ШИМ лазера
  
  document.getElementById('rotations').oninput = () => {
    updateDisplay();
    loadExactAngle();
  };
  document.getElementById('revTime').oninput = updateDisplay;
  document.getElementById('pauseTime').oninput = updateDisplay;
  document.getElementById('direction').onchange = saveSettings;
  document.getElementById('laserPwm').oninput = queueLaserPwmUpdate;
});
</script>

</body>
</html>
)rawliteral";

  server.send(200, "text/html", html);
}

// GET /move — один поворот стола. Ответ "OK" после завершения поворота.
// Клиент (скрипт записи кадров) должен после получения OK подождать 0.5 сек и затем сделать снимок.
void handleMove() {
  ESP.wdtFeed();  // Сброс watchdog перед началом обработки
  
  revTime = server.arg("revTime").toFloat();
  int count = server.hasArg("count") ? server.arg("count").toInt() : positionsPerTurn;
  if (count < 1) count = 1;
  positionsPerTurn = count;
  // Явно парсим dir: 1 = CW, 0 = CCW (toInt() надёжнее сравнения строк)
  direction = (server.arg("dir").toInt() == 1);
 
  // Всегда одно и то же число шагов — все повороты на одинаковый угол
  performRotationSteps(getStepsPerTurnByCount(count), revTime, direction);
  
  ESP.wdtFeed();  // Сброс watchdog после выполнения

  server.send(200, "text/plain", "OK");
}

void handleSequence() {
  if (isRunning) {
    server.send(409, "text/plain", "Sequence already running");
    return;
  }
  
  // Отправляем ответ сразу, чтобы не блокировать клиент
  server.send(200, "text/plain", "Sequence started");
  
  int count = server.arg("count").toInt();
  float revTime = server.arg("revTime").toFloat();
  float pauseTime = server.arg("pauseTime").toFloat();
  bool dir = (server.arg("dir").toInt() == 1);  // 1 = CW, 0 = CCW
  
  if (count < 1) count = 1;
  positionsPerTurn = count;
  if (revTime < 0.1) revTime = 0.1;
  if (pauseTime < 0) pauseTime = 0;
  
  isRunning = true;
  
  Serial.print("Запуск последовательности: ");
  Serial.print(count);
  Serial.print(" поворотов, время поворота: ");
  Serial.print(revTime);
  Serial.print(" сек, задержка: ");
  Serial.print(pauseTime);
  Serial.println(" сек");
  
  // Один и тот же шаг на каждом повороте (без распределения и округлений)
  int stepsThisTurn = getStepsPerTurnByCount(count);

  for (int i = 0; i < count && isRunning; i++) {
    Serial.print("Поворот ");
    Serial.print(i + 1);
    Serial.print(" из ");
    Serial.println(count);
    
    // Выполняем один поворот фиксированным числом шагов
    performRotationSteps(stepsThisTurn, revTime, dir);
    
    // Задержка перед следующим поворотом (кроме последнего)
    if (i < count - 1 && isRunning && pauseTime > 0) {
      unsigned long pauseStart = millis();
      while ((millis() - pauseStart) < (pauseTime * 1000) && isRunning) {
        server.handleClient();
        yield();
        delay(10);
      }
    }
  }
  
  isRunning = false;
  Serial.println("Последовательность завершена");
}

void handleStop() {
  isRunning = false;
  server.send(200, "text/plain", "Stopped");
  Serial.println("Последовательность остановлена пользователем");
}

void handleAngle() {
  // Возвращает точный угол одного поворота в градусах (для отображения на странице)
  int count = server.hasArg("count") ? server.arg("count").toInt() : positionsPerTurn;
  if (count < 1) count = 1;
  float angle = getExactAngleDegreesByCount(count);
  server.send(200, "text/plain", String(angle, 4));
}

void performRotationSteps(int stepsToMove, float rotationTime, bool dir) {
  if (stepsToMove < 1) return;
  direction = dir;
  
  speedDelay = (rotationTime * 1000000.0) / (stepsToMove * 2);
  if (speedDelay < 100) speedDelay = 100;
  
  stepMotor(stepsToMove);
  
  if (direction) {
    currentPosition += stepsToMove;
  } else {
    currentPosition -= stepsToMove;
  }
  
  ESP.wdtFeed();
}

void performSingleRotation(float rotationTime, bool dir) {
  direction = dir;
  
  // Один и тот же угол для каждого поворота — фиксированное число шагов
  int stepsToMove = getStepsPerFixedAngle();
  
  performRotationSteps(stepsToMove, rotationTime, dir);
}

void handleSetMicrostep() {
  if (server.hasArg("mode")) {
    int mode = server.arg("mode").toInt();
    if (mode >= 0 && mode <= 3) {
      setMicrostepMode(mode);
      server.send(200, "text/plain", "OK");
    } else {
      server.send(400, "text/plain", "Invalid mode");
    }
  } else {
    server.send(400, "text/plain", "Missing mode parameter");
  }
}

void setup() {
  Serial.begin(115200);
  
  // Увеличиваем timeout watchdog для предотвращения перезагрузки
  ESP.wdtDisable();
  ESP.wdtEnable(8000);  // 8 секунд вместо стандартных 3

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, HIGH);  // выключить двигатель по умолчанию (HIGH = off для TMC2209)
  pinMode(LASER_PIN, OUTPUT);
  analogWriteRange(1023);
  analogWriteFreq(1000);
  // Принудительно выключаем лазер при загрузке
  setLaser(false);
  
  // Настройка режима микрошагов TMC2209 (MS1, MS2)
  if (MS1_PIN >= 0) {
    pinMode(MS1_PIN, OUTPUT);
  }
  if (MS2_PIN >= 0) {
    pinMode(MS2_PIN, OUTPUT);
  }
  
  if (MS1_PIN >= 0 && MS2_PIN >= 0) {
    setMicrostepMode(microstepMode);
  } else {
    Serial.print("MS1/MS2 не подключены к ESP8266. Используем FIXED_MICROSTEP_MODE=");
    Serial.println(microstepMode);
    Serial.print("Расчёт выполнен для ");
    Serial.print(getStepsPerRev());
    Serial.println(" шагов/оборот.");
    Serial.println("Если реальный режим драйвера другой, угол поворота будет неверным.");
  }

  Serial.print("Позиций за оборот: ");
  Serial.println(positionsPerTurn);
  Serial.print("Шагов на один шаг стола: ");
  Serial.println(getStepsPerFixedAngle());
  Serial.print("Угол одного шага: ");
  Serial.print(getExactAngleDegrees(), 4);
  Serial.println("°");
  Serial.println("Serial: MOVE [time] [dir] | MOVE_DEG <angle_deg> [time] [dir], ответ OK");

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/move", handleMove);
  server.on("/sequence", handleSequence);
  server.on("/stop", handleStop);
  server.on("/angle", handleAngle);
  server.on("/laser", handleLaser);
  server.on("/setMicrostep", handleSetMicrostep);
  server.begin();
}

// Команды по Serial:
//   MOVE           — один поворот с текущими revTime и direction (фикс. шаг по POSITIONS_PER_TURN)
//   MOVE 1.2       — один поворот за 1.2 сек
//   MOVE 1.2 1     — время 1.2 сек, направление 1=CW 0=CCW
//   MOVE_DEG 120   — поворот на 120° (по умолчанию revTime и direction)
//   MOVE_DEG 120 1.2 1  — угол 120°, время 1.2 сек, направление 1=CW 0=CCW
void processSerialCommand() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;

  // Управление лазерной подсветкой:
  //   LASER_ON  / LASER 1  — включить лазер, ответ OK
  //   LASER_OFF / LASER 0  — выключить лазер, ответ OK
  if (cmd == "LASER_ON" || cmd == "LASER 1") {
    setLaser(true);
    Serial.println("OK");
    return;
  }
  if (cmd == "LASER_OFF" || cmd == "LASER 0") {
    setLaser(false);
    Serial.println("OK");
    return;
  }
  //   LASER_PWM <0..1023> — установить мощность через PWM, ответ OK
  if (cmd.startsWith("LASER_PWM ")) {
    String value = cmd.substring(String("LASER_PWM ").length());
    value.trim();
    setLaserPwm(value.toInt());
    Serial.println("OK");
    return;
  }

  // Управление удержанием мотора:
  //   HOLD_ON / HOLD 1 / H1  — включить удержание (EN=LOW), ответ OK
  //   HOLD_OFF / HOLD 0 / H0 — выключить удержание (EN=HIGH), ответ OK
  //   STOP / S               — синоним HOLD_OFF, ответ OK
  if (cmd == "HOLD_ON" || cmd == "HOLD 1" || cmd == "H1") {
    holdEnabled = true;
    digitalWrite(ENABLE_PIN, LOW);
    Serial.println("OK");
    return;
  }
  if (cmd == "HOLD_OFF" || cmd == "HOLD 0" || cmd == "H0" || cmd == "STOP" || cmd == "S") {
    holdEnabled = false;
    digitalWrite(ENABLE_PIN, HIGH);
    Serial.println("OK");
    return;
  }

  // MOVE_DEG <angle_deg> [time_sec] [dir] — поворот на угол в градусах
  if (cmd.startsWith("MOVE_DEG ") || cmd.startsWith("MOVE_D ")) {
    int sp = cmd.indexOf(' ');
    String rest = cmd.substring(sp + 1);
    rest.trim();
    float angleDeg = rest.toFloat();
    float t = revTime;
    bool d = direction;
    int sp2 = rest.indexOf(' ');
    if (sp2 > 0) {
      String rest2 = rest.substring(sp2 + 1);
      rest2.trim();
      t = rest2.toFloat();
      int sp3 = rest2.indexOf(' ');
      if (sp3 > 0) {
        d = (rest2.substring(sp3 + 1).toInt() == 1);
      }
      if (t < 0.1f) t = 0.1f;
      if (t > 60.0f) t = 60.0f;
    }
    if (angleDeg < 0.01f) angleDeg = 0.01f;
    int steps = (int)((angleDeg / 360.0f) * (float)getStepsPerRev() + 0.5f);
    if (steps < 1) steps = 1;
    ESP.wdtFeed();
    performRotationSteps(steps, t, d);
    Serial.println("OK");
    ESP.wdtFeed();
    return;
  }

  bool isMove = (cmd == "MOVE" || cmd == "M" || cmd == "1" ||
                 cmd.startsWith("MOVE ") || cmd.startsWith("M "));
  if (!isMove) return;

  float t = revTime;
  bool d = direction;
  int sp = cmd.indexOf(' ');
  if (sp >= 0) {
    String rest = cmd.substring(sp + 1);
    rest.trim();
    int sp2 = rest.indexOf(' ');
    if (sp2 > 0) {
      t = rest.substring(0, sp2).toFloat();
      d = (rest.substring(sp2 + 1).toInt() == 1);
    } else {
      t = rest.toFloat();
    }
    if (t < 0.1f) t = 0.1f;
    if (t > 60.0f) t = 60.0f;
    revTime = t;
    direction = d;
  }

  ESP.wdtFeed();
  performRotationSteps(getStepsPerFixedAngle(), revTime, direction);
  Serial.println("OK");
  ESP.wdtFeed();
}

void loop() {
  processSerialCommand();
  server.handleClient();
}
