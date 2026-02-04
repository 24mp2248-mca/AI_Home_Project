class AIChatbot {
    constructor(options = {}) {
        this.pageContext = options.context || {};
        this.apiEndpoint = 'http://localhost:8000/chat/';
        this.isOpen = false;

        this.init();
    }

    init() {
        // Inject HTML
        const container = document.createElement('div');
        container.innerHTML = `
            <button class="chat-fab" id="chatFab">💬</button>
            <div class="chat-window" id="chatWindow">
                <div class="chat-header">
                    <h4>AI Consultant</h4>
                    <button class="chat-close" id="chatClose">×</button>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="chat-msg bot">
                        Hello! 👋 I'm your AI design consultant. I can help you with cost estimates, design advice, or gardening inspiration. How can I help you today?
                    </div>
                </div>
                <div class="chat-input-area">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Ask about cost, colors, garden...">
                    <button class="chat-send" id="chatSend">➤</button>
                </div>
            </div>
        `;
        document.body.appendChild(container);

        // Elements
        this.fab = document.getElementById('chatFab');
        this.window = document.getElementById('chatWindow');
        this.closeBtn = document.getElementById('chatClose');
        this.messages = document.getElementById('chatMessages');
        this.input = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('chatSend');

        // Styles
        this.loadStyles();

        // Events
        this.fab.addEventListener('click', () => this.toggleChat());
        this.closeBtn.addEventListener('click', () => this.toggleChat());
        this.sendBtn.addEventListener('click', () => this.handleSend());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSend();
        });
    }

    loadStyles() {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'chatbot.css';
        document.head.appendChild(link);
    }

    toggleChat() {
        this.isOpen = !this.isOpen;
        this.window.classList.toggle('open', this.isOpen);
        if (this.isOpen) this.input.focus();
    }

    async handleSend() {
        const text = this.input.value.trim();
        if (!text) return;

        // User Msg
        this.appendMessage(text, 'user');
        this.input.value = '';

        // Typing indicator
        const loadingId = this.appendMessage('...', 'bot');

        try {
            // Context enrichment (grab latest global vars if available)
            const context = { ...this.pageContext };

            // If on Vis page, try to get room count dynamically
            if (window.houseLayout) {
                context.total_area = window.houseLayout.totalWidth * window.houseLayout.totalDepth; // Rough Est
            }

            // API Call
            const res = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, context: context })
            });

            const data = await res.json();

            // Remove typing bubble
            const loader = document.getElementById(loadingId);
            if (loader) loader.remove();

            // Bot Response
            this.appendMessage(data.text, 'bot', data.action);

        } catch (e) {
            console.error(e);
            const loader = document.getElementById(loadingId);
            if (loader) loader.remove();
            this.appendMessage("Sorry, I'm having trouble connecting to the brain.", 'bot');
        }
    }

    appendMessage(text, sender, action = null) {
        const id = 'msg-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
        const div = document.createElement('div');
        div.className = `chat-msg ${sender}`;
        div.id = id;
        div.textContent = text;

        if (action && sender === 'bot') {
            const btn = document.createElement('button');
            btn.className = 'chat-action-btn';
            btn.textContent = this.getActionLabel(action);
            btn.onclick = () => this.executeAction(action);
            div.appendChild(btn);
        }

        this.messages.appendChild(div);
        this.messages.scrollTop = this.messages.scrollHeight;
        return id;
    }

    getActionLabel(action) {
        if (action.type === 'change_color') return `Apply ${action.value} Paint`;
        if (action.type === 'add_feature') return `Add ${action.value}`;
        if (action.type === 'add_furniture') return `Place ${action.value}`;
        return 'Apply Changes';
    }

    executeAction(action) {
        // Dispatch generic event with full action details
        const event = new CustomEvent('ai-action', { detail: action });
        window.dispatchEvent(event);

        // Feedback
        if (action.type === 'change_color') {
            this.appendMessage(`Applying ${action.value} paint...`, 'bot');
        } else if (action.type === 'add_feature') {
            this.appendMessage(`Adding ${action.value} to the scene...`, 'bot');
        } else if (action.type === 'add_furniture') {
            this.appendMessage(`Placing ${action.value} in the center...`, 'bot');
        }
    }
}
