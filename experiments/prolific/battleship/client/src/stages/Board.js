export default class Board {
    constructor(size, tiles, ships) {
      this.size = size;
      this.tiles = tiles;
      this.ships = ships;
    }
  
    getTiles() {
      return this.tiles;
    }

    getTilesFlat() {
      return this.tiles.flat();
    }
  
    getShipLength(ship) {
      return this.getTilesFlat().filter(tile => tile == ship).length;
    }

}
  
export function hasShipSunk(occluded_board, true_board, ship) {
      var occ_length = occluded_board.getShipLength(ship);
      var true_length = true_board.getShipLength(ship);
      if (occ_length == true_length) {
          return true
      } else {
          return false
      }
}
  
export function hasGameEnded(occluded_board, true_board, ships) {
      var ships_sunk = ships.map((ship) => hasShipSunk(occluded_board, true_board, ship));
      var all_ships_sunk = ships_sunk.every(Boolean);
      if (all_ships_sunk) {
          return true
      } else {
          return false
      }
}
  