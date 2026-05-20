const { Puter } = require('puter.js');
const CustomProvider = require('./custom-provider');

const provider = new CustomProvider();
const puter = new Puter(provider);

puter.init().then(() => {
  console.log('Puter.js initialized with custom provider');
});