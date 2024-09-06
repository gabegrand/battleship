export function SendMessage(input, message_type, round, game, timer) {

    round.set(message_type,input);

    var stage_time = 0;
    if (game == "timeout") {
        stage_time = 0;
    } else {
        stage_time = game.get("roundDuration")*1000 - timer?.remaining;
    }

    var newMessage = new Map();
    newMessage.set("text", input);
    newMessage.set("type", message_type);
    newMessage.set("time", stage_time);

    var updated_messages = [...round.get("messages"), Object.fromEntries(newMessage)];
    round.set("messages", updated_messages);
}       

export function SendMessageIntro(input, message_type, player) {

    player.stage.set(message_type,input);

    var stage_time = 0;

    var newMessage = new Map();
    newMessage.set("text", input);
    newMessage.set("type", message_type);
    newMessage.set("time", stage_time);

    var updated_messages = [...player.stage.get("messages"), Object.fromEntries(newMessage)];
    player.stage.set("messages", updated_messages);
}    

export default SendMessage;
