module.exports = {
  apps: [{
    name: 'gateway',
    script: '/tmp/start_gateway.sh',
    restart_delay: 15000,
    max_restarts: 100,
    kill_timeout: 15000,
    listen_timeout: 60000,
    autorestart: true,
  }]
};
