#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <FastLED.h>

#define STEP_PIN D2
#define DIR_PIN  D3
#define ENABLE_PIN D1  // пин для включения/выключения двигателя (перенесен с D4)
#define LED_PIN D4     // пин для адресного светодиода (WS2812B/NeoPixel) - D4
// Если нужно использовать D4 для светодиода, ENABLE_PIN перенесен на D1
// Свободные пины: D1 (GPIO5), D4 (GPIO2), D5 (GPIO14), D6 (GPIO12), D7 (GPIO13), D8 (GPIO15)
#define NUM_LEDS 1      // количество адресных светодиодов
// Пины микрошагов для DRV8825
// 
// ВАРИАНТ 1: Управление через ESP8266 (программно)
// DRV8825 M0 -> ESP8266 D5 (GPIO14)
// DRV8825 M1 -> ESP8266 D6 (GPIO12)
// DRV8825 M2 -> ESP8266 D7 (GPIO13)
//
// ВАРИАНТ 2: Фиксированное подключение через резисторы (без программного управления)
// Для ПОЛНОГО ШАГА (200 шагов/оборот) - все пины должны быть LOW:
//   DRV8825 M0 -> GND (через pull-down 10 кОм или напрямую)
//   DRV8825 M1 -> GND (через pull-down 10 кОм или напрямую)
//   DRV8825 M2 -> GND (через pull-down 10 кОм или напрямую)
//
// Для установки HIGH (если нужен другой режим):
//   DRV8825 M0 -> VCC (питание логики 3.3V/5V) через pull-up 10 кОм
//   DRV8825 M1 -> VCC через pull-up 10 кОм
//   DRV8825 M2 -> VCC через pull-up 10 кОм
//
// ПРИМЕЧАНИЕ: Для полного шага нужны LOW, не HIGH!
//
#define M0_PIN -1  // установите -1, если используете фиксированное подключение
#define M1_PIN -1  // установите -1, если используете фиксированное подключение
#define M2_PIN -1  // установите -1, если используете фиксированное подключение

const char* ssid = "Cudy-EFB4";
const char* password = "MYpass-1";

ESP8266WebServer server(80);

int speedDelay = 2000;   // микросекунды между шагами (рассчитывается автоматически)
float revTime = 1.2f;  // время одного шага в секундах (по умолчанию; меньше = быстрее)
bool direction = true;
bool isRunning = false;  // флаг выполнения последовательности
int microstepMode = 5;  // режим микрошагов: 0=полный, 1=1/2, 2=1/4, 3=1/8, 4=1/16, 5=1/32 (1/32 по резисторам)

// Число шагов (позиций) за полный оборот 360°. Должно делить getStepsPerRev(), чтобы мотор
// останавливался в полном микрошаге. Для 1/32 (6400 шагов/об): 8, 10, 16, 20, 25, 32...
// Около 12 не подходит (6400/12 не целое). Выбрано 10 — 36° на шаг, стабильная остановка.
const int POSITIONS_PER_TURN = 10;
long currentPosition = 0;

// Массив для адресных светодиодов
CRGB leds[NUM_LEDS];

// Функция для получения количества шагов на оборот в зависимости от режима
int getStepsPerRev() {
  switch(microstepMode) {
    case 0: return 200;   // полный шаг
    case 1: return 400;   // 1/2 шага
    case 2: return 800;   // 1/4 шага
    case 3: return 1600;  // 1/8 шага
    case 4: return 3200;  // 1/16 шага
    case 5: return 6400;  // 1/32 шага
    default: return 200;
  }
}

// Функция для получения количества микрошагов в одном полном шаге
int getMicrostepsPerFullStep() {
  switch(microstepMode) {
    case 0: return 1;    // полный шаг
    case 1: return 2;    // 1/2 шага
    case 2: return 4;    // 1/4 шага
    case 3: return 8;    // 1/8 шага
    case 4: return 16;   // 1/16 шага
    case 5: return 32;   // 1/32 шага
    default: return 1;
  }
}

// Шагов на один шаг стола (одна позиция). Целое число — мотор полностью останавливается.
int getStepsPerFixedAngle() {
  int spr = getStepsPerRev();
  if (spr <= 0 || POSITIONS_PER_TURN <= 0) return 0;
  return spr / POSITIONS_PER_TURN;
}

// Точный угол одного поворота в градусах (шаги / шаги_на_оборот * 360)
float getExactAngleDegrees() {
  int spr = getStepsPerRev();
  int steps = getStepsPerFixedAngle();
  if (spr <= 0) return 0.0f;
  return (float)steps * 360.0f / (float)spr;
}

// Функция для установки режима микрошагов
void setMicrostepMode(int mode) {
  if (M0_PIN < 0 || M1_PIN < 0 || M2_PIN < 0) {
    Serial.println("ВНИМАНИЕ: Пины M0, M1, M2 не подключены к ESP8266!");
    Serial.println("Настройте режим микрошагов вручную через перемычки на модуле.");
    return;
  }
  
  microstepMode = mode;
  bool m0, m1, m2;
  
  switch(mode) {
    case 0: m0=0; m1=0; m2=0; break;  // полный шаг
    case 1: m0=1; m1=0; m2=0; break;  // 1/2 шага
    case 2: m0=0; m1=1; m2=0; break;  // 1/4 шага
    case 3: m0=1; m1=1; m2=0; break;  // 1/8 шага
    case 4: m0=0; m1=0; m2=1; break;  // 1/16 шага
    case 5: m0=1; m1=0; m2=1; break;  // 1/32 шага
    default: m0=0; m1=0; m2=0; break;
  }
  
  digitalWrite(M0_PIN, m0);
  digitalWrite(M1_PIN, m1);
  digitalWrite(M2_PIN, m2);
  
  Serial.print("Режим микрошагов установлен: ");
  Serial.print(mode);
  Serial.print(" (");
  Serial.print(getStepsPerRev());
  Serial.println(" шагов/оборот)");
}

void stepMotor(int steps) {
  // Включаем красный светодиод - индикатор движения
  leds[0] = CRGB::Red;  // красный цвет
  FastLED.show();
  
  // Сначала устанавливаем направление ДО включения двигателя
  digitalWrite(DIR_PIN, direction);
  delayMicroseconds(10);  // задержка для установки направления (минимум 5 мкс для DRV8825)
  
  // Включаем двигатель после установки направления
  digitalWrite(ENABLE_PIN, LOW);  // включить двигатель (LOW = включен для DRV8825)
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
  digitalWrite(ENABLE_PIN, HIGH);  // выключить двигатель (отключить удержание)
  
  // Выключаем светодиод - движение завершено
  leds[0] = CRGB::Black;  // выключить
  FastLED.show();
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
  // dir: 1 = CW, 0 = CCW (явно передаём число как строку)
  fetch('/move?revTime=' + encodeURIComponent(t) + '&dir=' + (dir ? '1' : '0'));
}

// Загрузка точного угла с платы
function loadExactAngle() {
  fetch('/angle')
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
  
  document.getElementById('rotations').oninput = updateDisplay;
  document.getElementById('revTime').oninput = updateDisplay;
  document.getElementById('pauseTime').oninput = updateDisplay;
  document.getElementById('direction').onchange = saveSettings;
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
  // Явно парсим dir: 1 = CW, 0 = CCW (toInt() надёжнее сравнения строк)
  direction = (server.arg("dir").toInt() == 1);
 
  // Всегда одно и то же число шагов — все повороты на одинаковый угол
  performRotationSteps(getStepsPerFixedAngle(), revTime, direction);
  
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
  int stepsThisTurn = getStepsPerFixedAngle();

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
  // Выключаем светодиод после завершения
  leds[0] = CRGB::Black;
  FastLED.show();
  Serial.println("Последовательность завершена");
}

void handleStop() {
  isRunning = false;
  // Выключаем светодиод при остановке
  leds[0] = CRGB::Black;
  FastLED.show();
  server.send(200, "text/plain", "Stopped");
  Serial.println("Последовательность остановлена пользователем");
}

void handleAngle() {
  // Возвращает точный угол одного поворота в градусах (для отображения на странице)
  float angle = getExactAngleDegrees();
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
    if (mode >= 0 && mode <= 5) {
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
  digitalWrite(ENABLE_PIN, HIGH);  // выключить двигатель по умолчанию
  
  // Инициализация адресных светодиодов
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(255);  // яркость (0-255), можно настроить
  
  // Принудительно выключаем светодиод (D4 может быть активен при загрузке)
  leds[0] = CRGB::Black;  // выключить светодиод
  FastLED.show();
  delay(10);  // небольшая задержка
  leds[0] = CRGB::Black;  // повторно выключаем для надежности
  FastLED.show();
  
  // Настройка режима микрошагов (M0, M1, M2)
  // Если пины подключены к ESP8266, укажите номера пинов в начале файла
  if (M0_PIN >= 0) {
    pinMode(M0_PIN, OUTPUT);
  }
  if (M1_PIN >= 0) {
    pinMode(M1_PIN, OUTPUT);
  }
  if (M2_PIN >= 0) {
    pinMode(M2_PIN, OUTPUT);
  }
  
  // Режим микрошагов установлен физически через резисторы (1/32 шага)
  // Если пины подключены к ESP8266, можно управлять программно
  if (M0_PIN >= 0 && M1_PIN >= 0 && M2_PIN >= 0) {
    // Пины подключены - устанавливаем режим программно
    setMicrostepMode(microstepMode);  // используем значение по умолчанию (1/32 шага)
  } else {
    // Пины не подключены - режим установлен физически
    Serial.print("Режим микрошагов установлен физически: 1/32 шага (");
    Serial.print(getStepsPerRev());
    Serial.println(" шагов/оборот)");
    Serial.println("Пины M0, M1, M2 не подключены к ESP8266 - управление через резисторы.");
  }

  Serial.print("Позиций за оборот: ");
  Serial.println(POSITIONS_PER_TURN);
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
